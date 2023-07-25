import logging

import streamlit as st

from src.common import rate_batch
from src.general_app import run_general
from src.recipe_app import run_recipe

logging.basicConfig(level=logging.INFO)


def main():
    accounts = dict(map(lambda ac: ac.split(':')[::-1], st.secrets['accounts'].split(' ')))

    access_code = st.text_input('Access Code', type='password', help='You need an access code to enter.')

    if access_code.startswith('rate'):
        # recipe feedback
        rate_batch(access_code=access_code)
    else:
        if access_code not in accounts:
            st.info("Please enter a valid access code to proceed.")
        else:
            account = accounts[access_code]
            if account == 'Touch':
                st.info(f"Logged in as {account}.")
                run_general(account=account)
            elif account == 'Jane':
                st.info(f"Logged in as {account}.")
                run_recipe(account=account)
            elif account == 'Chris':
                st.info(f"Logged in as {account}.")
                run_general(account=account)


if __name__ == '__main__':
    main()
