Accompanying code for the DMRN+21 submission "Differentiable Kirchhoff Carrier Stiff String" to the [Digital Music Research Network One-day Workshop 2026](https://www.qmul.ac.uk/dmrn/dmrn21/), scheduled for Friday, 11 December 2026. The implementation is based on [Russo, Ducceschi & Bilbao (2026)](https://github.com/Nemus-Project/nLinStringsConv-SAV/blob/main/KCstring/KCStringSAV_noSplit.m)'s `KCStringSAV_nosplit.m` with `SAV = false`.

## Installation

```sh
# macOS / Linux
curl -fsSL https://pixi.sh/install.sh | sh
export PATH="$HOME/.pixi/bin:$PATH"
git submodule update --init --recursive
pixi install

# Windows PowerShell alternative (run separately):
# powershell -ExecutionPolicy Bypass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```