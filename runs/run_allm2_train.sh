#!/bin/bash
cd /data/lib/podman-data/projects/goml
export GOTORCH_LIBS_DIR=/data/lib/podman-data/projects/gotorch/v6/libs
go test -run 'TestALLM_TrainStep_SignOfLife_F32Recon' -timeout 5m -v ./internal/abjexam/ 2>&1
