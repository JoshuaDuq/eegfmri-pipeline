#!/usr/bin/env bash

append_optional_memory_arg() {
    local array_name="$1"
    local memory="$2"
    local -n arguments="${array_name}"

    if [[ -n "${memory}" ]]; then
        arguments+=("--mem=${memory}")
    fi
}
