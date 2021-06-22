"""Utility methods"""

import os


def check_path(path, warn_exists=True, require_exists=False):

    """Check path to directory.
        warn_exists: Warns and requires validation by user to use the specified path if it already exists.
        require_exists: Aborts if the path does not exist. """

    if path[-1] != '/':
        path = path + '/'

    create_path = True

    if os.path.exists(path):
        create_path = False
        if warn_exists:
            replace = ''
            while replace not in ['y', 'n']:
                replace = input(f"Path {path} already exists. Files may be replaced. Continue? (y/n): ")
                if replace == 'y':
                    pass
                elif replace == 'n':
                    exit('Aborting, run again with a different path.')
                else:
                    print("Invalid input")


    if require_exists:
        if not os.path.exists(path):
            exit(f"{path} does not exist. Aborting")

    if create_path:
        os.mkdir(path)
        print(f"Created {path}")

    return path
