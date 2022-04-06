#!/bin/bash

# Get the absolute location of this script 
# https://stackoverflow.com/questions/59895/how-can-i-get-the-source-directory-of-a-bash-script-from-within-the-script-itsel
SOURCE=${BASH_SOURCE[0]}
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
  SOURCE=$(readlink "$SOURCE")
  [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )

# Location of active git hooks 
HOOKS="${DIR}/../.git/hooks"

# Print diff of new hooks
for hook_inactive_path in $(find "${DIR}" -type f \( ! -name "*.*" \)); do 
	hook=$(basename "$hook_inactive_path")
	hook_active_path="${HOOKS}/${hook}"

	if [[ -f "$hook_active_path" ]]; then # Hook already active 
		if ! cmp --silent "$hook_inactive_path" "$hook_active_path"; then # Hook changed
			echo "################################################################################"
			echo "# Changed hook '$hook'"
			echo "################################################################################"
			diff "$hook_inactive_path" "$hook_active_path"
			echo ""
		else # No changes -> skip file 
			continue 
		fi
	else # New hook
		echo "################################################################################"
		echo "# New hook '$hook'"
		echo "################################################################################"
		cat "$hook_inactive_path"
		echo ""
	fi

	echo "Do you want these changes to become active?"
	select answer in "Yes" "No"; do 
		case $answer in
			Yes) 
				cp "$hook_inactive_path" "$hook_active_path"
				break;;
			No)
				break;;
		esac
	done
done

