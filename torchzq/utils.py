def message_box(title, content, width=None):
    width = width or max(map(len, [title, *content.splitlines()])) + 8

    nb = width - 2  # number of blanks
    border = f'│{{: ^{nb}}}│'

    out = []
    out.append('┌' + '─' * nb + '┐')
    out.append(border.format(title.capitalize()))
    out.append('├' + '─' * nb + '┤')

    for line in content.splitlines():
        out.append(border.replace('^', '<').format(line.strip()))

    out.append('└' + '─' * nb + '┘')

    return '\n'.join(out)
