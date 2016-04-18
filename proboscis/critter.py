import inspect
import click
from click.core import Command
class ApplicativeCommand(Command):
    def __init__(self, c: Command):
        super().__init__(
            name=c.name,
            context_settings=c.context_settings,
            callback=c.callback,
            params=c.params,
            help=c.help,
            epilog=c.epilog,
            short_help=c.short_help,
            options_metavar=c.options_metavar,
            add_help_option=c.add_help_option
        )

    def map(self, f):
        return ApplicativeCommand(map_command(self, f))

    def __add__(self, other):
        return ApplicativeCommand(add_command(self, other, name="added command"))

    def __or__(self, other):
        return ApplicativeCommand(map_command(self, lambda p: other.map(lambda f: f(p))))


def add_command(a: Command, b: Command, name=None) -> Command:
    def merged(**kwargs):
        a_p = extract_params(a.params, a.params + b.params, kwargs)
        b_p = extract_params(b.params, a.params + b.params, kwargs)
        return (a.callback(**a_p), b.callback(**b_p))

    return Command(
        name=name,
        params=a.params + b.params,
        callback=merged
    )

def map2_command(a:Command,b:Command,f)->Command:
    def merged(**kwargs):
        a_p = extract_params(a.params, a.params + b.params, kwargs)
        b_p = extract_params(b.params, a.params + b.params, kwargs)
        return f(a.callback(**a_p), b.callback(**b_p))

    return Command(
        name=a.name+b.name,
        params=a.params + b.params,
        callback=merged
    )

def map_command(command: Command, f) -> Command:
    import copy
    new_command = copy.copy(command)
    new_command.callback = lambda **kwargs: f(command.callback(**kwargs))
    return new_command


def extract_params(params: list, whole_params: list, kwargs: dict):
    t_params = {s.name: kwargs[s.name] for s in params}
    return t_params


def command(f):
    return ApplicativeCommand(click.command()(f))


@command
@click.argument("a")
@click.option("--get", help="set a for this cmd")
def test(a, get="ant"):
    """
    :param a:
    :param get:
    :return:
    """
    return a


@command
@click.argument("b")
@click.option("--set", help="set b for this cmd")
def test2(b, set="ant"):
    """
    :param a:
    :param get:
    :return:
    """
    return b


if __name__ == '__main__':
    (test + test2).map(lambda t: print(t[1] + "," + t[0]))()
