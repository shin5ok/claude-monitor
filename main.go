// claude-usage: Claude サブスクリプションの利用状況をリアルタイム表示する TUI。
//
// セッション(5時間)と週間の利用率(%)、リセットまでの残り時間を、
// グラフィカルなバーと数字で表示し、一定間隔で自動更新する。
//
// データ取得元は Claude Code が /usage で使う OAuth エンドポイント:
//
//	GET https://api.anthropic.com/api/oauth/usage
//
// 認証は Claude Code の OAuth アクセストークンを利用する。macOS では Keychain
// (サービス名 "Claude Code-credentials") を優先し、無ければ ~/.claude/.credentials.json
// にフォールバックする。トークンは毎回読み直すため、Claude Code が
// リフレッシュすれば追従する。
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"
)

const usageURL = "https://api.anthropic.com/api/oauth/usage"

// ---- 認証情報 ----

type credentials struct {
	ClaudeAiOauth struct {
		AccessToken      string `json:"accessToken"`
		SubscriptionType string `json:"subscriptionType"`
		RateLimitTier    string `json:"rateLimitTier"`
	} `json:"claudeAiOauth"`
}

// keychainService は macOS Keychain に保存される Claude Code の項目名。
const keychainService = "Claude Code-credentials"

// readFromKeychain は macOS Keychain から認証情報 JSON を取り出す。
// security コマンド経由で読むため CGo 依存はなし。darwin 以外では使えない。
func readFromKeychain() ([]byte, error) {
	if runtime.GOOS != "darwin" {
		return nil, fmt.Errorf("Keychain は macOS のみサポート")
	}
	out, err := exec.Command("security", "find-generic-password",
		"-s", keychainService, "-w").Output()
	if err != nil {
		return nil, fmt.Errorf("Keychain 読み込み失敗 (%s): %w", keychainService, err)
	}
	return bytes.TrimSpace(out), nil
}

// readFromFile は ~/.claude/.credentials.json などのファイルから JSON を読む。
func readFromFile(path string) ([]byte, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("ファイル読み込み失敗 (%s): %w", path, err)
	}
	return b, nil
}

// parseCredentials は credentials JSON を構造体に展開し、最低限の妥当性を確認する。
func parseCredentials(data []byte) (credentials, error) {
	var c credentials
	if err := json.Unmarshal(data, &c); err != nil {
		return c, fmt.Errorf("認証情報の JSON 解析に失敗: %w", err)
	}
	if c.ClaudeAiOauth.AccessToken == "" {
		return c, fmt.Errorf("accessToken が空です。Claude Code でログインしてください")
	}
	return c, nil
}

// loadToken は Claude Code の認証情報を取得する。
// macOS では Keychain を優先し、失敗時はファイルにフォールバックする。
// その他の OS ではファイルのみを参照する。
func loadToken(path string) (credentials, error) {
	var errs []error
	if runtime.GOOS == "darwin" {
		if data, err := readFromKeychain(); err == nil {
			return parseCredentials(data)
		} else {
			errs = append(errs, err)
		}
	}
	data, err := readFromFile(path)
	if err != nil {
		errs = append(errs, err)
		return credentials{}, fmt.Errorf("認証情報を読めません。Claude Code を起動してログインしてください: %w", errors.Join(errs...))
	}
	return parseCredentials(data)
}

// ---- API レスポンス ----

type bucket struct {
	Utilization float64 `json:"utilization"`
	ResetsAt    string  `json:"resets_at"`
}

type usageResp struct {
	FiveHour       *bucket `json:"five_hour"`
	SevenDay       *bucket `json:"seven_day"`
	SevenDayOpus   *bucket `json:"seven_day_opus"`
	SevenDaySonnet *bucket `json:"seven_day_sonnet"`
}

// リトライ設定。一過性のエラー (429 / 408 / 409 / 5xx / 接続エラー) のみ、
// 指数バックオフで最大 maxRetries 回まで再試行する。401 等はリトライしない。
const (
	maxRetries  = 3
	baseBackoff = 2 * time.Second
	maxBackoff  = 30 * time.Second
)

// transient はそのステータスがリトライ対象かを返す。
func transient(status int) bool {
	switch {
	case status == http.StatusTooManyRequests, // 429
		status == http.StatusRequestTimeout, // 408
		status == http.StatusConflict,       // 409
		status >= 500:                       // サーバ側エラー
		return true
	default:
		return false
	}
}

// retryAfter は Retry-After ヘッダ (秒数 or HTTP-date) を解釈する。無ければ 0。
func retryAfter(h http.Header, now time.Time) time.Duration {
	v := strings.TrimSpace(h.Get("Retry-After"))
	if v == "" {
		return 0
	}
	if secs, err := time.ParseDuration(v + "s"); err == nil {
		return secs
	}
	if t, err := http.ParseTime(v); err == nil {
		if d := t.Sub(now); d > 0 {
			return d
		}
	}
	return 0
}

// sleepCtx は ctx をキャンセル可能な待機。中断されたら ctx.Err() を返す。
func sleepCtx(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func fetchUsage(ctx context.Context, token string) (*usageResp, error) {
	backoff := baseBackoff
	var lastErr error

	for attempt := 0; attempt <= maxRetries; attempt++ {
		if attempt > 0 {
			if err := sleepCtx(ctx, backoff); err != nil {
				return nil, err
			}
			backoff *= 2
			if backoff > maxBackoff {
				backoff = maxBackoff
			}
		}

		u, retry, wait, err := tryFetchUsage(ctx, token)
		if err == nil {
			return u, nil
		}
		lastErr = err
		if !retry {
			return nil, err // 認証エラー等は即座に返す
		}
		// Retry-After が示す待機時間を優先 (バックオフより長ければ採用)。
		if wait > backoff {
			backoff = wait
			if backoff > maxBackoff {
				backoff = maxBackoff
			}
		}
	}
	return nil, fmt.Errorf("リトライ上限 (%d回) に到達: %w", maxRetries, lastErr)
}

// tryFetchUsage は 1 回分の取得を試みる。
// retry=true はリトライ可能なエラーであることを示し、wait は Retry-After 由来の推奨待機時間。
func tryFetchUsage(ctx context.Context, token string) (u *usageResp, retry bool, wait time.Duration, err error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, usageURL, nil)
	if err != nil {
		return nil, false, 0, err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("anthropic-beta", "oauth-2025-04-20")
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		// 接続エラーは一過性とみなしてリトライ。
		return nil, true, 0, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))

	if resp.StatusCode != http.StatusOK {
		if resp.StatusCode == http.StatusUnauthorized {
			return nil, false, 0, fmt.Errorf("認証エラー(401)。トークンが期限切れの可能性。Claude Code を一度起動してトークンを更新してください")
		}
		if resp.StatusCode == http.StatusTooManyRequests {
			wait = retryAfter(resp.Header, time.Now())
			return nil, true, wait, fmt.Errorf("レート制限(429)。再試行します")
		}
		if transient(resp.StatusCode) {
			return nil, true, retryAfter(resp.Header, time.Now()), fmt.Errorf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
		}
		return nil, false, 0, fmt.Errorf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	var parsed usageResp
	if err := json.Unmarshal(body, &parsed); err != nil {
		return nil, false, 0, fmt.Errorf("レスポンス解析に失敗: %w", err)
	}
	return &parsed, false, 0, nil
}

// ---- 描画 ----

const (
	reset  = "\033[0m"
	bold   = "\033[1m"
	dim    = "\033[2m"
	green  = "\033[32m"
	yellow = "\033[33m"
	red    = "\033[31m"
	cyan   = "\033[36m"
	gray   = "\033[90m"
)

// 全角文字を 2 幅として数えた表示幅でラベルを右パディングする。
func padLabel(s string, width int) string {
	w := 0
	for _, r := range s {
		if r >= 0x1100 && (r <= 0x115F || // ハングル
			(r >= 0x2E80 && r <= 0xA4CF) || // CJK 各種
			(r >= 0xAC00 && r <= 0xD7A3) || // ハングル音節
			(r >= 0xF900 && r <= 0xFAFF) || // CJK 互換漢字
			(r >= 0xFF00 && r <= 0xFF60) || // 全角英数
			(r >= 0xFFE0 && r <= 0xFFE6)) {
			w += 2
		} else {
			w++
		}
	}
	if w < width {
		s += strings.Repeat(" ", width-w)
	}
	return s
}

// 利用率に応じた色 (低: 緑 / 中: 黄 / 高: 赤)。
func colorFor(pct float64) string {
	switch {
	case pct >= 80:
		return red
	case pct >= 50:
		return yellow
	default:
		return green
	}
}

const (
	barWidth = 20 // バーの横幅 (コンパクト表示)
	lineW    = 50 // 区切り線・全体の目安幅
)

// グラフィカルなバーを生成する。
func bar(pct float64) string {
	if pct < 0 {
		pct = 0
	}
	if pct > 100 {
		pct = 100
	}
	filled := int(pct/100*barWidth + 0.5)
	col := colorFor(pct)
	return col + strings.Repeat("█", filled) + gray + strings.Repeat("░", barWidth-filled) + reset
}

// リセットまでの残り時間を人間向けに整形する。
func untilReset(resetsAt string, now time.Time) string {
	if resetsAt == "" {
		return "—"
	}
	t, err := time.Parse(time.RFC3339, resetsAt)
	if err != nil {
		return "?"
	}
	d := t.Sub(now)
	if d <= 0 {
		return "まもなくリセット"
	}
	days := int(d.Hours()) / 24
	hours := int(d.Hours()) % 24
	mins := int(d.Minutes()) % 60
	secs := int(d.Seconds()) % 60
	switch {
	case days > 0:
		return fmt.Sprintf("%dd %02dh %02dm", days, hours, mins)
	case hours > 0:
		return fmt.Sprintf("%dh %02dm %02ds", hours, mins, secs)
	default:
		return fmt.Sprintf("%dm %02ds", mins, secs)
	}
}

// 1 行分のメーター行を描く。横幅を抑えるため
//
//	ラベル / バー / 利用率 と、その下に淡色で「↺ 残り時間 (絶対時刻)」の 2 行構成。
func meterLine(label string, b *bucket, now time.Time) string {
	const labelW = 14
	if b == nil {
		return fmt.Sprintf("  %s%s%s %s(データなし)%s", bold, padLabel(label, labelW), reset, dim, reset)
	}
	col := colorFor(b.Utilization)
	localReset := ""
	if t, err := time.Parse(time.RFC3339, b.ResetsAt); err == nil {
		localReset = t.Local().Format("01/02 15:04")
	}
	return fmt.Sprintf("  %s%s%s %s %s%5.1f%%%s\n  %s%s↺ %s (%s)%s",
		bold, padLabel(label, labelW), reset,
		bar(b.Utilization),
		col, b.Utilization, reset,
		strings.Repeat(" ", labelW), gray, untilReset(b.ResetsAt, now), localReset, reset,
	)
}

func render(u *usageResp, cred credentials, interval time.Duration, lastErr error, all bool) {
	now := time.Now()
	var sb strings.Builder

	// 画面クリア + カーソル先頭へ。
	sb.WriteString("\033[H\033[2J")

	sb.WriteString(fmt.Sprintf("%s%s Claude 利用状況%s %s[%s/%s]%s\n",
		bold, cyan, reset, dim, cred.ClaudeAiOauth.SubscriptionType, cred.ClaudeAiOauth.RateLimitTier, reset))
	sb.WriteString(gray + strings.Repeat("─", lineW) + reset + "\n\n")

	if lastErr != nil {
		sb.WriteString(fmt.Sprintf("  %sエラー: %v%s\n\n", red, lastErr, reset))
	}

	if u != nil {
		sb.WriteString(meterLine("セッション(5h)", u.FiveHour, now) + "\n\n")
		// --all 指定時のみ週間枠を表示。未指定時はセッション(5H)のみ。
		if all {
			sb.WriteString(meterLine("週間(全体)", u.SevenDay, now) + "\n\n")
			// モデル別の週間枠は提供されている場合のみ表示。
			if u.SevenDayOpus != nil {
				sb.WriteString(meterLine("週間(Opus)", u.SevenDayOpus, now) + "\n\n")
			}
			if u.SevenDaySonnet != nil {
				sb.WriteString(meterLine("週間(Sonnet)", u.SevenDaySonnet, now) + "\n\n")
			}
		}
	}

	sb.WriteString(gray + strings.Repeat("─", lineW) + reset + "\n")
	sb.WriteString(fmt.Sprintf("%s 更新 %s (%ds間隔) Ctrl-C で終了%s\n",
		gray, now.Format("15:04:05"), int(interval.Seconds()), reset))

	fmt.Print(sb.String())
}

func main() {
	defHome, _ := os.UserHomeDir()
	defCreds := filepath.Join(defHome, ".claude", ".credentials.json")

	interval := flag.Duration("interval", 3*time.Minute, "更新間隔 (例: 3m, 5m)。レート制限回避のため最低 30s")
	credsPath := flag.String("creds", defCreds, "認証情報ファイルのパス")
	once := flag.Bool("once", false, "一度だけ取得して終了 (バー無しの簡易出力)")
	all := flag.Bool("all", false, "全ての枠を表示 (未指定時はセッション(5H)のみ)")
	flag.Parse()

	// レート制限を避けるため、ポーリング間隔の下限を 30 秒に固定する。
	if *interval < 30*time.Second {
		*interval = 30 * time.Second
	}

	// --once: スクリプト向けの単発出力。
	if *once {
		cred, err := loadToken(*credsPath)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		u, err := fetchUsage(ctx, cred.ClaudeAiOauth.AccessToken)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		now := time.Now()
		if u.FiveHour != nil {
			fmt.Printf("session: %.1f%% (reset in %s)\n", u.FiveHour.Utilization, untilReset(u.FiveHour.ResetsAt, now))
		}
		if u.SevenDay != nil {
			fmt.Printf("weekly:  %.1f%% (reset in %s)\n", u.SevenDay.Utilization, untilReset(u.SevenDay.ResetsAt, now))
		}
		return
	}

	// Ctrl-C で抜けたらカーソルを表示し直す。
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	fmt.Print("\033[?25l")       // カーソル非表示
	defer fmt.Print("\033[?25h") // 復帰

	var lastUsage *usageResp
	var lastErr error

	poll := func() {
		cred, err := loadToken(*credsPath)
		if err != nil {
			lastErr = err
			render(lastUsage, cred, *interval, lastErr, *all)
			return
		}
		c, cancel := context.WithTimeout(ctx, 15*time.Second)
		u, ferr := fetchUsage(c, cred.ClaudeAiOauth.AccessToken)
		cancel()
		if ferr != nil {
			lastErr = ferr
		} else {
			lastUsage = u
			lastErr = nil
		}
		render(lastUsage, cred, *interval, lastErr, *all)
	}

	poll() // 初回即時表示

	dataTicker := time.NewTicker(*interval)
	defer dataTicker.Stop()
	// カウントダウン表示を滑らかにするため 1 秒ごとに再描画する。
	clockTicker := time.NewTicker(time.Second)
	defer clockTicker.Stop()

	for {
		select {
		case <-ctx.Done():
			fmt.Print("\033[?25h\n")
			return
		case <-dataTicker.C:
			poll()
		case <-clockTicker.C:
			// API は叩かず、残り時間だけ更新。
			cred, _ := loadToken(*credsPath)
			render(lastUsage, cred, *interval, lastErr, *all)
		}
	}
}
