#!/bin/sh

# --batch: evita modo interactivo
# --yes: sobrescribe si existe
# --passphrase: usa la variable de entorno
# --output: GUARDA el resultado en un archivo en lugar de imprimirlo

gpg --quiet --batch --yes --decrypt --passphrase="$GPG_PASSPHRASE" \
--output release.json keyfile.json.gpg