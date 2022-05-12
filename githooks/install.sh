#!/bin/bash

# Get the absolute location of this script 
# https://stackoverflow.com/questions/59895/how-can-i-get-the-source-directory-of-a-bash-script-from-within-the-script-itsel
SOURCE=${BASH_SOURCE[0]}
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
  SOURCE=$(readlink "$SOURCE")
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )

# Location of active git hooks 
HOOKS="${DIR}/../.git/hooks"

# Iterate over all hooks and ask the user whether to install it if it changed
# We consider all files without file extension in the same directory as this script to be git hooks
find "$DIR" -type f \( ! -name "*.*" \) -exec bash -c '
	hook_inactive_path="$1"
	hook=$(basename "$hook_inactive_path")
	hook_active_path="'"$HOOKS"'/${hook}"

	if [[ -f "$hook_active_path" ]]; then # Hook is already installed
		if ! cmp --silent "$hook_inactive_path" "$hook_active_path"; then # Hook changed
			echo "################################################################################"
			echo "# Changed hook $hook"
			echo "################################################################################"
			diff "$hook_inactive_path" "$hook_active_path"
			echo ""
		else # No changes -> skip file 
			exit 
		fi
	else # New hook
		echo "################################################################################"
		echo "# New hook $hook"
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
' bash {} \;
