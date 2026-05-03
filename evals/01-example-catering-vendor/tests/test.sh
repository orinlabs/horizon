#!/usr/bin/env bash
set -u

mkdir -p /logs/verifier

if python /tests/judge.py; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
