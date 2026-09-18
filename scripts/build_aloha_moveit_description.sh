#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
    echo "usage: $0 OUTPUT_DIRECTORY" >&2
    exit 2
fi

readonly output_directory="$1"
readonly interbotix_root=/opt/interbotix_ros_manipulators/interbotix_ros_xsarms
readonly description_root="${interbotix_root}/interbotix_xsarm_descriptions"
readonly moveit_root="${interbotix_root}/interbotix_xsarm_moveit"
readonly source_commit=b66d5b905725351dd71d3251a06cd3f4c777940f
readonly source_archive_sha256=d22c67bf76a83de275e547f07ed9959bbf5a4335fe0da4ff092efa7094ab7637
readonly official_urdf_sha256=45dff1e0de2456386dbb64b851c94db2d874b6dbefb6503cb747bc55435177bd
readonly official_srdf_sha256=39658212772f0432398e61d6b05b3bfbfac059c7ef7e3b12b5df584e9c76493b

if [[ -e "${output_directory}" ]]; then
    echo "refusing to overwrite existing output: ${output_directory}" >&2
    exit 1
fi
mkdir -p "${output_directory}/expanded" "${output_directory}/generated"

# shellcheck disable=SC1091
set +u
source /opt/ros/humble/setup.bash
set -u

echo "${official_urdf_sha256}  ${description_root}/urdf/aloha_vx300s.urdf.xacro" \
    | sha256sum --check --strict
echo "${official_srdf_sha256}  ${moveit_root}/config/srdf/vx300s.srdf.xacro" \
    | sha256sum --check --strict

for side in left right; do
    xacro \
        "${description_root}/urdf/aloha_vx300s.urdf.xacro" \
        "robot_name:=vx300s_${side}" \
        use_world_frame:=false \
        hardware_type:=actual \
        > "${output_directory}/expanded/${side}.urdf"
    xacro \
        "${moveit_root}/config/srdf/vx300s.srdf.xacro" \
        "robot_name:=vx300s_${side}" \
        > "${output_directory}/expanded/${side}.srdf"
done

python3 /opt/rosetta/scripts/compose_aloha_moveit_description.py \
    --left-urdf "${output_directory}/expanded/left.urdf" \
    --right-urdf "${output_directory}/expanded/right.urdf" \
    --left-srdf "${output_directory}/expanded/left.srdf" \
    --right-srdf "${output_directory}/expanded/right.srdf" \
    --output-urdf "${output_directory}/generated/aloha_bimanual.urdf" \
    --output-srdf "${output_directory}/generated/aloha_bimanual.srdf" \
    --output-manifest "${output_directory}/generated/description_manifest.json" \
    --source-commit "${source_commit}" \
    --source-archive-sha256 "${source_archive_sha256}"

check_urdf "${output_directory}/generated/aloha_bimanual.urdf"
