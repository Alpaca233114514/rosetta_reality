#!/usr/bin/env bash
# Repackage existing, dpkg-verified Debian stdlib for offline synthetic tests.
set -Eeuo pipefail
cd /mnt/c/Users/Logan/Documents/GitHub/rosetta_reality
context=.cache/qualification-stdlib-image-20260927-001
test ! -e "$context"
mkdir -p "$context/rootfs/usr/bin" "$context/rootfs/usr/lib"
dpkg-query -W python3.13-minimal libpython3.13-stdlib libc6 > "$context/packages.txt"
dpkg -V python3.13-minimal libpython3.13-stdlib > "$context/package-verification.txt"
test ! -s "$context/package-verification.txt"
cp -L /usr/bin/python3.13 "$context/rootfs/usr/bin/"
cp -a /usr/lib/python3.13 "$context/rootfs/usr/lib/"
ldd /usr/bin/python3.13 /usr/lib/python3.13/lib-dynload/*.so > "$context/ldd.txt"
awk '/=> \// {print $3} /^\t\// {print $1}' "$context/ldd.txt" | sort -u > "$context/libraries.txt"
while IFS= read -r library; do
    destination="$context/rootfs$(dirname "$library")"
    mkdir -p "$destination"
    cp -L "$library" "$destination/"
done < "$context/libraries.txt"
tar -C "$context/rootfs" -cf "$context/rootfs.tar" .
sha256sum "$context/rootfs.tar" > "$context/rootfs.sha256"
printf 'FROM scratch\nADD rootfs.tar /\nENTRYPOINT ["/usr/bin/python3.13"]\n' > "$context/Dockerfile"
docker.exe build --network none --pull=false -t codex-local-qualification-stdlib:20260927-001 "$(wslpath -w "$(realpath "$context")")" 2>&1 | tee "$context/build.log"
docker.exe image inspect codex-local-qualification-stdlib:20260927-001 --format '{{.Id}}' > "$context/image-id.txt"
