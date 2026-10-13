# nsc-codec — build / test / deploy (`nsc` compressor + archiver, optional `nsc-gui`)
PREFIX   ?= $(HOME)/.local/bin
NSC_HOME ?= $(HOME)/.nsc

.PHONY: all build test golden selftest install install-gui gui bench uninstall doctor clean help

all: build

build:                 ## Build the optimized release binary (target/release/nsc)
	cargo build --release

test:                  ## Run the test suite (engine, formats, models, golden files, archives)
	cargo test --release

golden:                ## Regenerate the format golden files (only after a versioned format change)
	UPDATE_GOLDEN=1 cargo test --release golden

bench: build           ## Benchmark nsc vs classical compressors on bench/corpus (level 5)
	./target/release/nsc bench bench/corpus --levels 5 --classical --report bench/results/latest.json

gui:                   ## Build the desktop app (gui/target/release/nsc-gui)
	cd gui && cargo build --release

selftest: build        ## Run the Stage A–E parity harness (needs artifacts/)
	./target/release/nsc selftest

install: build         ## Build + install `nsc` and seed the model registry (see install.sh)
	PREFIX="$(PREFIX)" NSC_HOME="$(NSC_HOME)" ./install.sh

install-gui: build     ## Same as install, plus the `nsc-gui` desktop app
	INSTALL_GUI=1 PREFIX="$(PREFIX)" NSC_HOME="$(NSC_HOME)" ./install.sh

uninstall:             ## Remove the installed binary (keeps the registry)
	rm -f "$(PREFIX)/nsc" "$(PREFIX)/nsc-gui"
	@echo "removed $(PREFIX)/nsc  (registry at $(NSC_HOME) left intact)"

doctor: build          ## Check registry + python/tok.py + fingerprints
	./target/release/nsc --home "$(NSC_HOME)" doctor --verify

clean:                 ## Remove build artifacts
	cargo clean

help:                  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
