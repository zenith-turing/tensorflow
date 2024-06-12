bazelisk build \
    --override_repository=bazel_vscode_compdb=/home/yinze/.vscode-server/extensions/galexite.bazel-cpp-tools-1.0.5/compdb/ \
    --aspects=@bazel_vscode_compdb//:aspects.bzl%compilation_database_aspect \
    --color=no \
    --show_result=2147483647 \
    --noshow_progress \
    --noshow_loading_progress \
    --output_groups=compdb_files,header_files \
    --build_event_json_file=/tmp/tmp-2700-FFWKgiLL530O \
    --action_env=BAZEL_CPP_TOOLS_TIMESTAMP=1713595990.163 \
    //tensorflow:tensorflow_cc


/home/yinze/.vscode-server/extensions/galexite.bazel-cpp-tools-1.0.5/compdb/postprocess.py \
    -s -b /tmp/tmp-2700-FFWKgiLL530O && rm /tmp/tmp-2700-FFWKgiLL530O