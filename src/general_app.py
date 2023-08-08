import streamlit as st
from bojaxns import BayesianOptimisation

from src.common import load_experiment, new_experiment, save_experiment, Experiment, display_parameter_space, \
    trial_section
from src.interfaces import ABInterface


def run_general(account: str, ab_interface: ABInterface|None = None):
    # Load Experiment
    with st.sidebar.expander("Load Experiment"):
        load_experiment(account=account)

    # New Experiment
    with st.sidebar.expander("New Experiment"):
        new_experiment(account=account)

    if 'experiment' in st.session_state:
        if st.sidebar.button('Save Experiment') and ('experiment' in st.session_state):
            save_experiment(account=account, save_file='experiment_state.json',
                            experiment=st.session_state['experiment'])

    if 'experiment' not in st.session_state:
        st.info('Please load an experiment to proceed.')
        return

    experiment: Experiment = st.session_state['experiment']
    st.header(f'{experiment.experiment_name.title()}')
    st.markdown(f"### Rating system: {experiment.rating_system.title()}")
    with st.expander('Search Space'):
        display_parameter_space(parameter_space=experiment.opt_experiment.parameter_space)

    trial_section(experiment=experiment, ab_interface=ab_interface)

    try:
        bo_experiment = BayesianOptimisation(experiment=experiment.opt_experiment)
        fig = bo_experiment.visualise()
        st.header('Progress Visualisation')
        st.write(fig)
    except RuntimeError as e:
        if 'Nothing to visualise' in str(e):
            pass
        else:
            raise e
