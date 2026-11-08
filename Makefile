PREFIX ?= $(HOME)/.local
BIN     = $(PREFIX)/bin/localmodel

.PHONY: build test install clean

build:
	cargo build --release

test:
	cargo test

# Rebuild and replace the binary on PATH. Running `localmodel` from a stale
# ~/.local/bin copy silently loses every new feature, so make this the only
# supported way to publish a change.
install: build
	install -m 755 target/release/localmodel $(BIN)
	@echo "installed $(BIN)"

clean:
	cargo clean
