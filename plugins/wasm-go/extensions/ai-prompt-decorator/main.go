package main

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/higress-group/proxy-wasm-go-sdk/proxywasm"
	"github.com/higress-group/proxy-wasm-go-sdk/proxywasm/types"
	"github.com/higress-group/wasm-go/pkg/log"
	"github.com/higress-group/wasm-go/pkg/wrapper"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

func main() {}

func init() {
	wrapper.SetCtx(
		"ai-prompt-decorator",
		wrapper.ParseConfig(parseConfig),
		wrapper.ProcessRequestHeaders(onHttpRequestHeaders),
		wrapper.ProcessRequestBody(onHttpRequestBody),
	)
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ReplaceRule rewrites the content of messages matched by Role/Pattern.
// When OnRole is empty, the rule applies to messages of any role.
// When Regex is true, Pattern is compiled as a Go RE2 regular expression
// at config-parse time, and Replacement supports $1/$2 capture references.
// Otherwise the rule performs literal substring replacement.
type ReplaceRule struct {
	OnRole      string `json:"on_role,omitempty"`
	Pattern     string `json:"pattern"`
	Replacement string `json:"replacement"`
	Regex       bool   `json:"regex,omitempty"`

	compiled *regexp.Regexp `json:"-"`
}

type AIPromptDecoratorConfig struct {
	Prepend []Message     `json:"prepend"`
	Append  []Message     `json:"append"`
	Replace []ReplaceRule `json:"replace,omitempty"`
}

func parseConfig(jsonConfig gjson.Result, config *AIPromptDecoratorConfig) error {
	if err := json.Unmarshal([]byte(jsonConfig.Raw), config); err != nil {
		return err
	}
	for i := range config.Replace {
		rule := &config.Replace[i]
		if rule.Pattern == "" {
			return fmt.Errorf("replace[%d].pattern must not be empty", i)
		}
		if rule.Regex {
			re, err := regexp.Compile(rule.Pattern)
			if err != nil {
				return fmt.Errorf("replace[%d].pattern is not a valid regex: %w", i, err)
			}
			rule.compiled = re
		}
	}
	return nil
}

func onHttpRequestHeaders(ctx wrapper.HttpContext, config AIPromptDecoratorConfig) types.Action {
	ctx.DisableReroute()
	proxywasm.RemoveHttpRequestHeader("content-length")
	return types.ActionContinue
}

func replaceVariable(variable string, entry *Message) (*Message, error) {
	key := fmt.Sprintf("${%s}", variable)
	if strings.Contains(entry.Content, key) {
		value, err := proxywasm.GetProperty([]string{variable})
		if err != nil {
			return nil, err
		}
		entry.Content = strings.ReplaceAll(entry.Content, key, string(value))
	}
	return entry, nil
}

func decorateGeographicPrompt(entry *Message) (*Message, error) {
	geoArr := []string{"geo-country", "geo-province", "geo-city", "geo-isp"}

	var err error
	for _, geo := range geoArr {
		entry, err = replaceVariable(geo, entry)
		if err != nil {
			return nil, err
		}
	}

	return entry, nil
}

// applyReplaceRulesToContent applies all matching replace rules to a single
// message content string and returns the rewritten value. Rules are applied
// in declaration order so users get predictable layering when several rules
// could match the same role.
func applyReplaceRulesToContent(role, content string, rules []ReplaceRule) string {
	for _, rule := range rules {
		if rule.OnRole != "" && rule.OnRole != role {
			continue
		}
		if rule.Regex {
			if rule.compiled == nil {
				continue
			}
			content = rule.compiled.ReplaceAllString(content, rule.Replacement)
		} else {
			content = strings.ReplaceAll(content, rule.Pattern, rule.Replacement)
		}
	}
	return content
}

// applyReplaceRulesToMessage rewrites the "content" field of a JSON message
// in place when it is a plain string. Multimodal contents (arrays/objects)
// are returned untouched so we do not corrupt vision/audio payloads.
func applyReplaceRulesToMessage(rawMessage string, rules []ReplaceRule) string {
	if len(rules) == 0 {
		return rawMessage
	}
	role := gjson.Get(rawMessage, "role").String()
	contentResult := gjson.Get(rawMessage, "content")
	if contentResult.Type != gjson.String {
		return rawMessage
	}
	original := contentResult.String()
	updated := applyReplaceRulesToContent(role, original, rules)
	if updated == original {
		return rawMessage
	}
	out, err := sjson.Set(rawMessage, "content", updated)
	if err != nil {
		log.Errorf("Failed to apply replace rules to message, error: %v", err)
		return rawMessage
	}
	return out
}

// applyReplaceRulesToSystem rewrites Anthropic-style top-level "system"
// prompts. The field may be either a plain string or an array of content
// blocks; text blocks that become empty after replacement are dropped so
// dynamic attribution-only blocks do not remain in the cache prefix.
func applyReplaceRulesToSystem(rawSystem string, rules []ReplaceRule) string {
	if len(rules) == 0 {
		return rawSystem
	}
	systemResult := gjson.Parse(rawSystem)
	switch systemResult.Type {
	case gjson.String:
		original := systemResult.String()
		updated := applyReplaceRulesToContent("system", original, rules)
		if updated == original {
			return rawSystem
		}
		out, err := json.Marshal(updated)
		if err != nil {
			log.Errorf("Failed to apply replace rules to system prompt, error: %v", err)
			return rawSystem
		}
		return string(out)
	case gjson.JSON:
		if !systemResult.IsArray() {
			return rawSystem
		}
		systemJSON := `{"system":[]}`
		for _, block := range systemResult.Array() {
			rewritten := block.Raw
			textResult := block.Get("text")
			if textResult.Type == gjson.String {
				original := textResult.String()
				updated := applyReplaceRulesToContent("system", original, rules)
				if updated != original {
					if strings.TrimSpace(updated) == "" {
						continue
					}
					out, err := sjson.Set(block.Raw, "text", updated)
					if err != nil {
						log.Errorf("Failed to apply replace rules to system block, error: %v", err)
					} else {
						rewritten = out
					}
				}
			}
			systemJSON, _ = sjson.SetRaw(systemJSON, "system.-1", rewritten)
		}
		return gjson.Get(systemJSON, "system").Raw
	default:
		return rawSystem
	}
}

func onHttpRequestBody(ctx wrapper.HttpContext, config AIPromptDecoratorConfig, body []byte) types.Action {
	messageJson := `{"messages":[]}`

	for _, entry := range config.Prepend {
		entry, err := decorateGeographicPrompt(&entry)
		if err != nil {
			log.Errorf("Failed to decorate geographic prompt in prepend, error: %v", err)
			return types.ActionContinue
		}

		msg, err := json.Marshal(entry)
		if err != nil {
			log.Errorf("Failed to add prepend message, error: %v", err)
			return types.ActionContinue
		}
		rewritten := applyReplaceRulesToMessage(string(msg), config.Replace)
		messageJson, _ = sjson.SetRaw(messageJson, "messages.-1", rewritten)
	}

	rawMessage := gjson.GetBytes(body, "messages")
	if !rawMessage.Exists() {
		log.Errorf("Cannot find messages field in request body")
		return types.ActionContinue
	}
	for _, entry := range rawMessage.Array() {
		rewritten := applyReplaceRulesToMessage(entry.Raw, config.Replace)
		messageJson, _ = sjson.SetRaw(messageJson, "messages.-1", rewritten)
	}

	for _, entry := range config.Append {
		entry, err := decorateGeographicPrompt(&entry)
		if err != nil {
			log.Errorf("Failed to decorate geographic prompt in append, error: %v", err)
			return types.ActionContinue
		}

		msg, err := json.Marshal(entry)
		if err != nil {
			log.Errorf("Failed to add prepend message, error: %v", err)
			return types.ActionContinue
		}
		rewritten := applyReplaceRulesToMessage(string(msg), config.Replace)
		messageJson, _ = sjson.SetRaw(messageJson, "messages.-1", rewritten)
	}

	newbody, err := sjson.SetRaw(string(body), "messages", gjson.Get(messageJson, "messages").Raw)
	if err != nil {
		log.Error("modify body failed")
	}
	rawSystem := gjson.Get(newbody, "system")
	if rawSystem.Exists() {
		rewrittenSystem := applyReplaceRulesToSystem(rawSystem.Raw, config.Replace)
		if rewrittenSystem != rawSystem.Raw {
			newbody, err = sjson.SetRaw(newbody, "system", rewrittenSystem)
			if err != nil {
				log.Error("modify system prompt failed")
			}
		}
	}
	if err = proxywasm.ReplaceHttpRequestBody([]byte(newbody)); err != nil {
		log.Error("rewrite body failed")
	}

	return types.ActionContinue
}
