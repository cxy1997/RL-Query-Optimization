#!/bin/bash

. ./utils/gdownload.sh

echo "downloading jdk..."
gdownload 10ziyB7Vk3WYmv_OWR8KvhPClOe0tZsQ9 jdk.tar
echo "jdk downloaded"

echo "downloading jre..."
gdownload 16J7-fm_kmLlLgA5EMRZWnYcqES2aiNlR jre.tar
echo "jre downloaded"