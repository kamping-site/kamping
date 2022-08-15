# ----------------------------------
# Options affecting listfile parsing
# ----------------------------------
with section("parse"):
    # Specify structure for custom cmake functions
    additional_commands = {
        'katestophe_add_test_executable': {
            'flags': [],
            'kwargs': {'FILES': '*'}
        },
        'katestrophe_add_mpi_test': {
            'flags': ['DISCOVER_TESTS'],
            'kwargs': {'CORES': '*'}
        },
        'katestrophe_add_compilation_failure_test': {
            'flags': ['NO_EXCEPTION_MODE'],
            'kwargs': {'TARGET': '1', 'FILES': '*', 'SECTIONS': '*', 'LIBRARIES': '*'}
        },
        'kamping_register_test': {
            'flags': ['NO_EXCEPTION_MODE'],
            'kwargs': {'FILES': '*'}
        },
        'kamping_register_mpi_test': {
            'flags': ['NO_EXCEPTION_MODE'],
            'kwargs': {'FILES': '*', 'CORES': '*'}
        },
        'kamping_register_compilation_failure_test': {
            'flags': ['NO_EXCEPTION_MODE'],
            'kwargs': {'FILES': '*', 'SECTIONS': '*', 'LIBRARIES': '*'}
        },
    }

# -----------------------------
# Options affecting formatting.
# -----------------------------
with section("format"):

    # Disable formatting entirely, making cmake-format a no-op
    disable = False

    # How wide to allow formatted cmake files
    line_width = 120

    # How many spaces to tab for indent
    tab_size = 4

    # If true, lines are indented using tab characters (utf-8 0x09) instead of
    # <tab_size> space characters (utf-8 0x20).
    use_tabchars = False

    # If an argument group contains more than this many sub-groups (parg or kwarg
    # groups) then force it to a vertical layout.
    max_subgroups_hwrap = 2

    # If a positional argument group contains more than this many arguments, then
    # force it to a vertical layout.
    max_pargs_hwrap = 6

    # If a cmdline positional group consumes more than this many lines without
    # nesting, then invalidate the layout (and nest)
    max_rows_cmdline = 2

    # If true, separate flow control names from their parentheses with a space
    separate_ctrl_name_with_space = True

    # If true, separate function names from parentheses with a space
    separate_fn_name_with_space = False

    # If a statement is wrapped to more than one line, than dangle the closing
    # parenthesis on its own line.
    dangle_parens = True

    # If the trailing parenthesis must be 'dangled' on its on line, then align it
    # to this reference: `prefix`: the start of the statement,  `prefix-indent`:
    # the start of the statement, plus one indentation  level, `child`: align to
    # the column of the arguments
    dangle_align = 'prefix'

    # If the statement spelling length (including space and parenthesis) is
    # smaller than this amount, then force reject nested layouts.
    min_prefix_chars = 4

    # If the statement spelling length (including space and parenthesis) is larger
    # than the tab width by more than this amount, then force reject un-nested
    # layouts.
    max_prefix_chars = 10

    # If a candidate layout is wrapped horizontally but it exceeds this many
    # lines, then reject the layout.
    max_lines_hwrap = 2

    # What style line endings to use in the output.
    line_ending = 'unix'

    # Format command names consistently as 'lower' or 'upper' case
    command_case = 'canonical'

    # Format keywords consistently as 'lower' or 'upper' case
    keyword_case = 'unchanged'

    # A list of command names which should always be wrapped
    always_wrap = []

    # If true, the argument lists which are known to be sortable will be sorted
    # lexicographicall
    enable_sort = True

    # If true, the parsers may infer whether or not an argument list is sortable
    # (without annotation).
    autosort = False

    # By default, if cmake-format cannot successfully fit everything into the
    # desired linewidth it will apply the last, most agressive attempt that it
    # made. If this flag is True, however, cmake-format will print error, exit
    # with non-zero status code, and write-out nothing
    require_valid_layout = False

    # A dictionary mapping layout nodes to a list of wrap decisions. See the
    # documentation for more information.
    layout_passes = {}

# ------------------------------------------------
# Options affecting comment reflow and formatting.
# ------------------------------------------------
with section("markup"):

    # What character to use for bulleted lists
    bullet_char = '*'

    # What character to use as punctuation after numerals in an enumerated list
    enum_char = '.'

    # If comment markup is enabled, don't reflow the first comment block in each
    # listfile. Use this to preserve formatting of your copyright/license
    # statements.
    first_comment_is_literal = False

    # If comment markup is enabled, don't reflow any comment block which matches
    # this (regex) pattern. Default is `None` (disabled).
    literal_comment_pattern = None

    # Regular expression to match preformat fences in comments default=
    # ``r'^\s*([`~]{3}[`~]*)(.*)$'``
    fence_pattern = '^\\s*([`~]{3}[`~]*)(.*)$'

    # Regular expression to match rulers in comments default=
    # ``r'^\s*[^\w\s]{3}.*[^\w\s]{3}$'``
    ruler_pattern = '^\\s*[^\\w\\s]{3}.*[^\\w\\s]{3}$'

    # If a comment line matches starts with this pattern then it is explicitly a
    # trailing comment for the preceeding argument. Default is '#<'
    explicit_trailing_pattern = '#<'

    # If a comment line starts with at least this many consecutive hash
    # characters, then don't lstrip() them off. This allows for lazy hash rulers
    # where the first hash char is not separated by space
    hashruler_min_length = 10

    # If true, then insert a space between the first hash char and remaining hash
    # chars in a hash ruler, and normalize its length to fill the column
    canonicalize_hashrulers = True

    # enable comment markup parsing and reflow
    enable_markup = True
