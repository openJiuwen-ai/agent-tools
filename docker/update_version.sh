#!/bin/bash

VERSION=$1

echo "VERSION: ${VERSION}"

sed -i "s#version = \"[^\"]*\"#version = \"${VERSION}\"#g" marketplace/pyproject.toml