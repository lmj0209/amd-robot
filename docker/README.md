# Container status

`docker/Dockerfile` is a committed reproducibility specification that mirrors
`scripts/rgc_setup.sh`. It pins the measured package versions, applies the
reviewed Brax patch, preserves the gfx1100 environment, and defaults to the
fail-closed G0 smoke test.

The base image `amd-oneclick-base:rocm7.2.1-py3.12-v20260416` is available only
inside RGC. RGC did not expose its immutable registry digest, so this file is
not presented as a portable, digest-pinned public image. Use
`scripts/rgc_setup.sh` on the named RGC base as the primary reproduction path;
build this Dockerfile only on a runner that can resolve that image.
