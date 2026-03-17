package globalsetup

import "testing"

func TestDefaultConfigKeys_IncludesBidsRestRoot(t *testing.T) {
	keys := DefaultConfigKeys()
	found := false
	for _, key := range keys {
		if key == "paths.bids_rest_root" {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected paths.bids_rest_root in default config keys, got %#v", keys)
	}
}

func TestSetConfigValuesAndBuildOverrides_HandleBidsRestRoot(t *testing.T) {
	m := New(".")
	m.SetConfigValues(map[string]interface{}{
		"paths.bids_rest_root":  "/data/bids/rest",
		"paths.deriv_rest_root": "/data/derivatives/rest",
	})

	if m.bidsRestRoot != "/data/bids/rest" {
		t.Fatalf("expected bidsRestRoot to hydrate, got %q", m.bidsRestRoot)
	}
	if m.derivRestRoot != "/data/derivatives/rest" {
		t.Fatalf("expected derivRestRoot to hydrate, got %q", m.derivRestRoot)
	}

	m.bidsRestRoot = "/data/bids/rest-updated"
	m.derivRestRoot = "/data/derivatives/rest-updated"
	overrides := m.buildOverrides()
	paths, ok := overrides["paths"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected paths overrides, got %#v", overrides)
	}
	if paths["bids_rest_root"] != "/data/bids/rest-updated" {
		t.Fatalf("expected bids_rest_root override, got %#v", paths["bids_rest_root"])
	}
	if paths["deriv_rest_root"] != "/data/derivatives/rest-updated" {
		t.Fatalf("expected deriv_rest_root override, got %#v", paths["deriv_rest_root"])
	}
}
