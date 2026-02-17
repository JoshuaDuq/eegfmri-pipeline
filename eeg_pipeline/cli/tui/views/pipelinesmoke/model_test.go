package pipelinesmoke

import "testing"

func TestBuildCommand_IncludesSelectedPipelines(t *testing.T) {
	m := New("thermalactive")
	m.ToggleAll(false)
	m.ToggleByID("features")
	m.ToggleByID("plotting")

	cmd := m.BuildCommand()
	want := "scripts/tui_pipeline_smoke.py --task thermalactive --pipelines features,plotting"
	if cmd != want {
		t.Fatalf("expected %q, got %q", want, cmd)
	}
}
