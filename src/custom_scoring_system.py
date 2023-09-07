from typing import List, Set

from asteval import Interpreter

from src.interfaces import UserObservableResponse


def is_valid_expression(expression: str, allowed_vars: Set[str]):
    user_symbols = dict((key, 1.0) for key in allowed_vars)
    # Create an Interpreter with the allowed variables
    aeval = Interpreter(usersyms=user_symbols)
    try:
        # Try to evaluate the expression
        aeval(expression)
        return True
    except:
        return False


def create_function(expression: str):
    def func(**kwargs):
        aeval = Interpreter(usersyms=kwargs)
        return aeval(expression)

    return func


def get_all_observables(observables_list: List[UserObservableResponse]):
    all_observables = set()
    for observables in observables_list:
        for observable_resp in observables.observables:
            all_observables.add(observable_resp.observable_name)
    return all_observables
