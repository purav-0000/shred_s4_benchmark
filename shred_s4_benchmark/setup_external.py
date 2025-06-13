from shred_s4_benchmark.utils.external import install_repo

s4_repo = "shred_s4_benchmark/external/s4"
shred_repo = "shred_s4_benchmark/external/shred"

install_repo("https://github.com/state-spaces/s4", s4_repo)
install_repo("https://github.com/Jan-Williams/pyshred", shred_repo)