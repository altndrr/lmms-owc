#!/usr/bin/env bash
set -o errexit
set -o nounset
set -o pipefail

if [[ "${TRACE-0}" == "1" ]]; then
    set -o xtrace
fi

if [[ "${1-}" =~ ^-*h(elp)?$ ]]; then
    echo 'usage: interact.sh [-h]

Launch an iPython session with the project environment loaded.
'
    exit
fi

cd "$(dirname "$0")"
while [ "$(find . -maxdepth 1 -name pyproject.toml | wc -l)" -ne 1 ]; do cd ..; done

main() {
    uvx ipython --InteractiveShellApp.extra_extensions "autoreload" \
		--InteractiveShellApp.exec_lines "%autoreload 2" \
		--InteractiveShellApp.exec_lines "import dotenv; _ = dotenv.load_dotenv(dotenv.find_dotenv())" \
		--InteractiveShell.xmode "Context" \
		--no-banner --no-confirm-exit --pprint
}

main "$@"
