package execution

import (
	"reflect"
	"testing"
)

func TestCommandPartsPreferExplicitArgs(t *testing.T) {
	model := Model{
		Command:     `eeg-pipeline features --deriv-root 'broken`,
		CommandArgs: []string{"eeg-pipeline", "features", "--deriv-root", `C:\Users\Test User\derivatives`},
	}

	got := model.commandParts()
	want := []string{"eeg-pipeline", "features", "--deriv-root", `C:\Users\Test User\derivatives`}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("commandParts() = %#v, want %#v", got, want)
	}
}
