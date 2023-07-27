import base64
import string
from datetime import datetime
from random import choice
from typing import List, Dict, Literal
from uuid import uuid4

import streamlit as st
from bojaxns import OptimisationExperiment, Parameter, ContinuousPrior, IntegerPrior, CategoricalPrior, ParameterSpace, \
    NewExperimentRequest, BayesianOptimisation, TrialUpdate
from bojaxns.gaussian_process_formulation.distribution_math import NotEnoughData
from jax import random
from jax._src.random import PRNGKey
from pydantic import BaseModel, Field

from src.experiment_repo import S3Bucket, S3Interface


def bytes_to_str(b):
    return base64.b64encode(b).decode()


def str_to_bytes(s):
    return base64.b64decode(s)


def test_str_to_bytes():
    s = 'abcaflj24309upfijdsf'
    assert s == bytes_to_str(str_to_bytes(s))
    b = b'abcaflj24309upfijdsf'
    assert b == str_to_bytes(bytes_to_str(b))


class ParameterResponse(BaseModel):
    id: str
    name: str
    current_trial_id: str
    current_value: float | int
    updated_at: datetime
    is_active: bool


class ObservableResponse(BaseModel):
    parameter_id: str
    observable: float | int


class UserObservableResponse(BaseModel):
    user_id: str
    trial_id: str
    observables: List[ObservableResponse]


class PullResponse(BaseModel):
    current_parameters: List[ParameterResponse]
    user_observations: List[UserObservableResponse]


class ParameterRequest(BaseModel):
    parameter_name: str
    value: float | int


class PushRequest(BaseModel):
    trial_id: str
    parameters: List[ParameterRequest]


class NoTrialSet(Exception):
    def __init__(self, name: str):
        self.name = name
        super().__init__(f"No trial set for parameter: {name}")


def get_table_download_link(save_file):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    with open(save_file, 'r') as f:
        val = "\n".join(f.readlines())

    b64 = base64.b64encode(val.encode())  # .decode()
    # b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{save_file}">Download State File</a>'  # decode b'abc' => abc


class BatchMeta(BaseModel):
    name: str
    image: str | None = None


class Experiment(BaseModel):
    experiment_id: str
    experiment_name: str
    experimenter_account: str
    opt_experiment: OptimisationExperiment
    batch_meta: Dict[str, BatchMeta] = Field(
        default_factory=dict,
        description='map (trial) -> (batch_meta)'
    )
    rating_system: Literal['5 star system', 'percent system', 'unbounded system', 'culinary system']
    illegal_value: float


class FeedbackRef(BaseModel):
    experiment_id: str
    trial_id: str


class FeedbackMap(BaseModel):
    feedback_refs: Dict[str, FeedbackRef] = Field(
        default_factory=dict,
        description='Map (rate_code) -> (experiment, trial)'
    )


def get_feedback_map() -> FeedbackMap:
    s3_bucket = get_s3_bucket()
    with s3_bucket['feedback_map.json'] as f:
        if not f.exists():
            return FeedbackMap()
        return FeedbackMap.parse_raw(f.read())


def store_feedback_map(feedback_map: FeedbackMap):
    s3_bucket = get_s3_bucket()
    with s3_bucket['feedback_map.json'] as f:
        f.write(feedback_map.json(indent=2))


class ExperimentsMeta(BaseModel):
    experiment_name_map: Dict[str, str] = Field(
        default_factory=dict,
        description='Map (name) -> (id)'
    )


def get_experiments_meta(account: str) -> ExperimentsMeta:
    s3_bucket = get_s3_bucket()
    with s3_bucket[f'experiments_meta_{account}.json'] as f:
        if not f.exists():
            return ExperimentsMeta()
        return ExperimentsMeta.parse_raw(f.read())


def set_experiments_meta(account: str, experiments_meta: ExperimentsMeta):
    s3_bucket = get_s3_bucket()
    with s3_bucket[f'experiments_meta_{account}.json'] as f:
        f.write(experiments_meta.json(indent=2))


@st.cache_resource
def get_s3_bucket() -> S3Bucket:
    return S3Bucket(S3Interface(
        repo_name='app',
        aws_access_key_id=st.secrets['aws_access_key_id'],
        aws_secret_access_key=st.secrets['aws_secret_access_key']
    ))


def load_experiment(account: str):
    """
    Loads an experiment from file or online.

    Args:
        account: account name
    """
    source = st.radio('Source', options=['Online', 'Local'])
    if source == 'Local':
        file = st.file_uploader("Upload data file", type=['json'], accept_multiple_files=False,
                                help='If you have saved an experiment to your device, you can upload it here.')
        if st.button("Load file") and (file is not None):
            experiment = Experiment.parse_raw(file)
            st.info(f"Loaded {file}.")
            if 'experiment' in st.session_state:
                del st.session_state['experiment']
            st.session_state['experiment'] = experiment
    elif source == 'Online':
        experiments_meta = get_experiments_meta(account=account)
        names = list(experiments_meta.experiment_name_map.keys())

        if len(names) == 0:
            st.info('No experiments defined. Please create one.')
        else:
            experiment_name = st.selectbox("Experiments", options=names, key=f'select_experiment_{account}')
            do_delete = st.checkbox(f"I want to delete {experiment_name}.",
                                    key=f'do_experiment_delete_{experiment_name}')
            if do_delete:
                if st.button(f'Delete {experiment_name}'):
                    experiment_id = experiments_meta.experiment_name_map[experiment_name]
                    experiment = get_experiment(experiment_id=experiment_id)
                    if experiment is not None:
                        delete_experiment(experiment_id)
                    del experiments_meta.experiment_name_map[experiment_name]
                    set_experiments_meta(account=account, experiments_meta=experiments_meta)
                    st.experimental_rerun()
            if st.button("Load Experiment"):
                experiment_id = experiments_meta.experiment_name_map[experiment_name]
                experiment = get_experiment(experiment_id=experiment_id)
                if experiment is None:
                    st.info(f'Was unable to find {experiment_name}. Would you like to delete it?')
                    do_delete = st.checkbox(f"Yes I want to delete {experiment_name}.",
                                            key=f'do_experiment_delete_{experiment_name}')
                    if do_delete:
                        if st.button(f'Delete {experiment_name}'):
                            del experiments_meta.experiment_name_map[experiment_name]
                            set_experiments_meta(account=account, experiments_meta=experiments_meta)
                            st.experimental_rerun()
                else:
                    st.info(f"Loaded {experiment_name}.")
                    if 'experiment' in st.session_state:
                        del st.session_state['experiment']
                    st.session_state['experiment'] = experiment


def get_experiment(experiment_id: str) -> Experiment | None:
    s3_bucket = get_s3_bucket()
    with s3_bucket[f'experiment_{experiment_id}'] as f:
        if not f.exists():
            return None
        else:
            experiment = Experiment.parse_raw(f.read())
            return experiment


def delete_experiment(experiment_id: str):
    s3_bucket = get_s3_bucket()
    with s3_bucket[f'experiment_{experiment_id}'] as f:
        if f.exists():
            f.delete()


def copy_experiment(experiment: Experiment):
    def set_state(key: str, value):
        if key in st.session_state:
            del st.session_state[key]
        st.session_state[key] = value

    set_state('experiment_name', f"Copy of {experiment.experiment_name}")
    set_state('rating_system', experiment.rating_system.title())
    set_state('num_parameters', len(experiment.opt_experiment.parameter_space.parameters))

    for i, param in enumerate(experiment.opt_experiment.parameter_space.parameters):
        set_state(f'name_{i}', param.name)
        if isinstance(param.prior, ContinuousPrior):
            set_state(f'prior_type_{i}', 'Continuous')
            set_state(f'lower_{i}', float(param.prior.lower))
            set_state(f'upper_{i}', float(param.prior.upper))
            set_state(f'mode_{i}', float(param.prior.mode))
        elif isinstance(param.prior, IntegerPrior):
            set_state(f'prior_type_{i}', 'Integer')
            set_state(f'lower_{i}', int(param.prior.lower))
            set_state(f'upper_{i}', int(param.prior.upper))
            set_state(f'mode_{i}', float(param.prior.mode))
        elif isinstance(param.prior, CategoricalPrior):
            set_state(f'prior_type_{i}', 'Categorical')
            set_state(f'num_cat_{i}', len(param.prior.probs))
            for n in range(len(param.prior.probs)):
                set_state(f'p_{n}_{i}', param.prior.probs[n])


def new_experiment(account: str):
    if 'experiment' in st.session_state:
        experiment = st.session_state['experiment']
        if st.button(f'Duplicate "{experiment.experiment_name}"', key='duplicate_experiment',
                     help='You can duplicate experiment then tune it to a new experiment.'):
            copy_experiment(experiment=experiment)

    experiment_name = st.text_input("Experiment name", placeholder='e.g. Novel Buttertart', key='experiment_name',
                                    help='Pick a good name for your experiment. '
                                         'It should be unique so you do not confuse with another '
                                         'experiment of same name.')
    rating_system = st.selectbox("What type of rating system do you want to use?",
                                 options=['Culinary System',
                                          '5 Star System',
                                          'Percent System',
                                          'Unbounded System'],
                                 help='What type of system do you want to use for rating trials?',
                                 key='rating_system')
    if rating_system == 'Unbounded System':
        illegal_value = float('nan')
    elif rating_system == '5 Star System':
        illegal_value = -1
    elif rating_system == 'Percent System':
        illegal_value = -10
    elif rating_system == 'Culinary System':
        illegal_value = -0.1
    else:
        raise ValueError(f'invalid rating system {rating_system}')
    num_params = st.number_input(label='Number of Parameters', min_value=1, max_value=10, step=1,
                                 key='num_parameters',
                                 help="Choose the most important parameters that you want to learn about. "
                                      "It is always better to first apply all your expert knowledge, "
                                      "and isolate the parameters that you're least sure about.")
    init_explore_num = st.number_input(label='Number of initial explore points', min_value=2, step=1,
                                       key='num_init_explore',
                                       help='Initial explore points will choose some initial points to explore your '
                                            'parameter space, before it can start optimising.')
    parameters: List[Parameter] = []
    for i in range(num_params):
        st.subheader(f'Param #{i + 1}')
        param_name = st.text_input(label='Name', key=f'name_{i}',
                                   placeholder='e.g. vanilla (tsp), bake time (min), pecans (True/False)')
        prior_type = st.selectbox(label='Variable Type', index=0,
                                  options=['Continuous', 'Integer', 'Categorical'],
                                  key=f'prior_type_{i}')
        if prior_type is None:
            st.error("Select a prior type")
            continue
        if prior_type == 'Continuous':
            lower = st.number_input(label='Lower', value=0., key=f'lower_{i}')
            upper = st.number_input(label='Upper', value=1., key=f'upper_{i}')
            if lower >= upper:
                st.error("Lower must be < Upper")
            mode = st.number_input(label='Best Guess', key=f'mode_{i}', value=(upper + lower) / 2.,
                                   help='Best guess of location of optimum.')
            uncert = max(upper - lower, 1e-6)
            # uncert = st.number_input(label='uncert', min_value=0.01, key=f'uncert_{i}',
            #                          help='Uncertainty of location of optimum.')
            prior = ContinuousPrior(lower=lower, upper=upper, mode=mode, uncert=uncert)
        elif prior_type == 'Integer':
            lower = st.number_input(label='Lower', value=0, key=f'lower_{i}', step=1, help='inclusive')
            upper = st.number_input(label='Upper', value=1, key=f'upper_{i}', step=1, help='inclusive')
            if lower >= upper:
                st.error("Lower must be < Upper")
            mode = st.number_input(label='Best Guess', key=f'mode_{i}', value=(upper + lower) / 2.,
                                   help='Best guess of location of optimum.')
            uncert = max(upper - lower, 1e-6)
            # uncert = st.number_input(label='uncert', min_value=0.01, key=f'uncert_{i}',
            #                          help='Uncertainty of location of optimum.')
            prior = IntegerPrior(lower=lower, upper=upper, mode=mode, uncert=uncert)
        elif prior_type == 'Categorical':
            num_cat = st.number_input(label='Number of Categories', min_value=2, max_value=5, step=1,
                                      key=f'num_cat_{i}',
                                      help='Tip, if you want to explore decisions like Yes/No then use this with 2 categories.')
            probs = []
            for n in range(num_cat):
                p = st.number_input(label=f'P(cat_{n} | optimal)', min_value=0., max_value=1., value=1. / num_cat,
                                    key=f'p_{n}_{i}',
                                    help="Probability of each category being optimal (if not normalised, then we will normalise for you).")
                probs.append(p)
            prior = CategoricalPrior(probs=probs)
        else:
            raise ValueError(f"Invalid {prior_type}")
        param = Parameter(name=param_name, prior=prior)
        parameters.append(param)

    if st.button('Create New Experiment'):
        parameter_space = ParameterSpace(parameters=parameters)
        new_experiment_request = NewExperimentRequest(
            parameter_space=parameter_space,
            init_explore_size=init_explore_num
        )
        bo_experiment = BayesianOptimisation.create_new_experiment(new_experiment=new_experiment_request)
        experiment_id = str(uuid4())
        experiment = Experiment(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            experimenter_account=account,
            opt_experiment=bo_experiment.experiment,
            rating_system=rating_system.lower(),
            illegal_value=illegal_value
        )
        if 'experiment' in st.session_state:
            del st.session_state['experiment']
        st.session_state['experiment'] = experiment
        experiments_meta = get_experiments_meta(account=account)

        experiments_meta.experiment_name_map[experiment_name] = experiment_id
        set_experiments_meta(account=account, experiments_meta=experiments_meta)

        store_experiment(experiment=experiment)


def store_experiment(experiment: Experiment):
    with st.spinner("Storing..."):
        s3_bucket = get_s3_bucket()
        with s3_bucket[f'experiment_{experiment.experiment_id}'] as f:
            f.write(experiment.json(indent=2))


def save_experiment(account: str, save_file: str, experiment: Experiment):
    # experiments_meta = get_experiments_meta(account=account)
    #
    # experiments_meta.experiment_name_map[experiment.experiment_name] = experiment.experiment_id
    # set_experiments_meta(account=account, experiments_meta=experiments_meta)
    #
    # store_experiment(experiment=experiment)

    with open(save_file, 'w') as f:
        f.write(experiment.json(indent=2))
    st.sidebar.info(f"Saved experiment to {save_file}.")
    st.sidebar.markdown(get_table_download_link(save_file), unsafe_allow_html=True)


def display_parameter_space(parameter_space: ParameterSpace):
    for i in range(len(parameter_space.parameters)):
        param = parameter_space.parameters[i]
        if isinstance(param.prior, ContinuousPrior):
            st.markdown(f"### `{param.name}`\n"
                        f"* Continuous in [{param.prior.lower}, {param.prior.upper}]")
        elif isinstance(param.prior, IntegerPrior):
            st.markdown(f"### `{param.name}`\n"
                        f"* Integer in [{param.prior.lower}, {param.prior.upper}]")
        elif isinstance(param.prior, CategoricalPrior):
            st.markdown(f"### `{param.name}`\n"
                        f"* Categorical n={len(param.prior.probs)}")
        else:
            raise ValueError(f"Invalid {param.prior}")


def new_trial(key: PRNGKey, experiment: Experiment, beta: float, random_explore: bool):
    with st.spinner('... Creating trial'):
        opt_experiment = experiment.opt_experiment
        num_before = len(opt_experiment.trials)
        bo_experiment = BayesianOptimisation(experiment=opt_experiment)
        trial_id = bo_experiment.create_new_trial(
            key=key,
            random_explore=random_explore,
            beta=beta
        )
        num_after = len(opt_experiment.trials)
        if num_before == num_after:
            st.info("You already have some trials waiting to be run. "
                    "Please run those trials and report measurements before requesting a new one.")
            return
    st.info(f"Created a new trial: {trial_id}")
    store_experiment(experiment=experiment)


def display_trials(experiment: Experiment):
    opt_experiment = experiment.opt_experiment
    for trial_id in sorted(opt_experiment.trials, key=lambda ti: opt_experiment.trials[ti].create_dt):
        trial = opt_experiment.trials[trial_id]
        with st.expander(f"Trial: {trial.trial_id}"):
            st.markdown(f"**Created**: {trial.create_dt}")
            s = ""
            for name, value in trial.param_values.items():
                s += f"* `{name}` = {value.value}\n"
            st.markdown(s)
            bo_experiment = BayesianOptimisation(experiment=opt_experiment)

            action = st.radio('What would you like to do?',
                              options=["I want to delete this trial.",
                                       "I would like to generate a rater code.",
                                       "I would like to provide my own rating.",
                                       "I would like to mark this trial as illegal."
                                       ],
                              key=f'action_{trial_id}')
            if action == "I want to delete this trial.":
                if st.button("Delete Trial", key=f'delete_trial_{trial_id}'):
                    bo_experiment.delete_trial(trial_id=trial_id)
                    store_experiment(experiment=experiment)
                    st.experimental_rerun()
            elif action == "I would like to generate a rater code.":
                st.info("Great, you're ready to collect feedback!")
                batch_name = st.text_input("How would like your raters to see this batch?",
                                           placeholder='e.g. Banana Foster Tart',
                                           help='Your rater will see this name.',
                                           key=f'batch_name_{trial_id}')
                batch_meta = BatchMeta(name=batch_name, image=None)
                image = st.camera_input("Take a picture for your raters (optional)",
                                        key=f'batch_pic_{trial_id}',
                                        help='This can help your raters identify what they are rating, '
                                             'e.g. a flavour of butter tart.')
                if image is not None:
                    st.markdown("#### Preview how it'll look to your users:")
                    st.image(image, caption=batch_name)
                    batch_meta.image = bytes_to_str(image.getvalue())
                else:
                    if (trial_id in experiment.batch_meta) and (experiment.batch_meta[trial_id].image is not None):
                        st.image(str_to_bytes(experiment.batch_meta[trial_id].image),
                                 caption=experiment.batch_meta[trial_id].name)

                if st.button('Generate Rater Code',
                             help='Generate a new code for each rater. Each rater can only submit one rating.'):

                    experiment.batch_meta[trial_id] = batch_meta
                    store_experiment(experiment=experiment)

                    feedback_map = get_feedback_map()
                    feedback_ref = FeedbackRef(
                        experiment_id=experiment.experiment_id,
                        trial_id=trial_id
                    )
                    code = "".join(list(map(lambda i: choice(string.ascii_uppercase), range(6))))
                    rate_code = f"rate-{code}"
                    while rate_code in feedback_map.feedback_refs:
                        code = "".join(list(map(lambda i: choice(string.ascii_uppercase), range(6))))
                        rate_code = f"rate-{code}"
                    feedback_map.feedback_refs[rate_code] = feedback_ref
                    store_feedback_map(feedback_map)
                    if f'last_rate_code_{trial_id}' in st.session_state:
                        del st.session_state[f'last_rate_code_{trial_id}']
                    st.session_state[f'last_rate_code_{trial_id}'] = rate_code
                if f'last_rate_code_{trial_id}' in st.session_state:
                    st.markdown(f"Last Rater Code: `{st.session_state[f'last_rate_code_{trial_id}']}`")
            elif action == "I would like to provide my own rating.":
                st.info("Great, you'd like to provide your own rating feedback!")
                ref_id = st.text_input('Reference ID', value=f"{experiment.experimenter_account}",
                                       help="A unique reference for your rating.",
                                       key=f'ref_id_{trial.trial_id}')
                ask_rating(ref_id=ref_id, trial_id=trial.trial_id,
                           experiment=experiment)
            elif action == 'I would like to mark this trial as illegal.':
                st.info('This means you find this parameter combination so bad, it should not be considered. '
                        'This information is valuable, so do not delete this trial! Simply mark it as illegal.')
                ref_id = st.text_input('Reference ID', value=f"{experiment.experimenter_account}",
                                       help="A unique reference for your rating.",
                                       key=f'ref_id_{trial.trial_id}')
                obj_val = experiment.illegal_value

                if st.button('Report Illegal Trial', key=f'report_update_{trial_id}'):
                    bo_experiment = BayesianOptimisation(experiment=opt_experiment)
                    bo_experiment.post_measurement(
                        trial_id=trial.trial_id,
                        trial_update=TrialUpdate(ref_id=ref_id, objective_measurement=obj_val)
                    )
                    store_experiment(experiment=experiment)

            st.markdown(f"**Trial Updates**: {len(trial.trial_updates)}")
            s = ''
            for ref_id, trial_update in sorted(trial.trial_updates.items()):
                s += f"* `{ref_id}` -> {trial_update.objective_measurement} at {trial_update.measurement_dt}\n"
            st.markdown(s)


def trial_section(experiment: Experiment):
    st.markdown('### Search')
    key_seed = st.number_input('What is your favourite number?', min_value=0, max_value=2 ** 32 - 1, step=1, value=42,
                               help='Use to seed the random number generator.')
    exploration = st.slider('Exploration/Exploitation', min_value=0., max_value=1., value=1., key='explore_beta',
                            help='The closer to one, the more you will explore uncharted territory. '
                                 'The closer to zero, the more you will stop exploring and try to find the best. '
                                 'It is recommended to explore as long as possible, before starting to exploit.')
    random_explore = st.checkbox("I'm feeling lucky!", value=False, key='random_explore',
                                 help="If true, then you'll explore with a shot in the dark. "
                                      "Don't worry if you don't like your suggestion, "
                                      "you can delete it later and try again.")
    key = random.PRNGKey(key_seed)
    if st.button('Suggest next trial', key='create_trial',
                 help='This will use all your past data to suggest the next best parameter setting to try.'):
        try:
            new_trial(key=key, experiment=experiment, beta=exploration,
                      random_explore=random_explore)
        except NotEnoughData:
            st.info("You need more complete trials before we can learn structure in the data. "
                    "Use 'I'm feeling lucky!' to get another randomised trial.")
    st.markdown('### Trials')
    display_trials(experiment=experiment)


def ask_rating(ref_id: str, trial_id: str, experiment: Experiment):
    if experiment.rating_system == 'culinary system':
        rating_dict = {
            "Dreadful (0%): Absolutely inedible, you regret even trying it.": 0.01,
            "Terrible (18%): It leaves a bad taste in your mouth, figuratively and literally.": 0.216,
            "Poor (31%): It disappoints your tastebuds. A real letdown.": 0.379,
            "Subpar (41%): Unimpressive, it doesn't meet your expectations.": 0.509,
            "Mediocre (49%): Just okay, half of what you've tried is worse, half is better.": 0.612,
            "Acceptable (56%): It's alright, a satisfactory taste experience.": 0.694,
            "Fair (63%): Not bad at all, you can appreciate some aspects.": 0.759,
            "Good (68%): Pleasant, it has some commendable features.": 0.810,
            "Quite Good (73%): It impresses, you enjoy it more than you expected.": 0.852,
            "Very Good (77%): A tasty delight that leaves a positive impression.": 0.884,
            "Excellent (81%): Memorable for its taste, you're happily satisfied.": 0.910,
            "Impressive (84%): Beyond expectations, you'd happily have it again.": 0.931,
            "Superb (87%): A cut above, it's a real treat for your palate.": 0.947,
            "Outstanding (89%): Exceptional taste, you're already craving more.": 0.960,
            "Exquisite (91%): A stellar taste experience, you're running out of compliments.": 0.970,
            "Distinguished (94%): Deserves high praise for its superior taste.": 0.978,
            "Magnificent (95%) A flawless blend of flavors. Truly impressive.": 0.985,
            "Extraordinary (97%): A culinary masterpiece, it's tantalizingly close to perfect.": 0.990,
            "Almost Perfect (98%): So delicious it's hard to believe it's not perfect.": 0.994,
            "Nearly Flawless (99%): Just one tiny step away from perfection.": 0.997,
            "The Pinnacle (100%): As good as it can possibly be. The epitome of culinary delight.": 1.00
        }
        options = sorted(rating_dict.keys(), key=lambda k: rating_dict[k])
        rating_choice = st.select_slider('Please select the rating that best matches your thoughts.',
                                         options=options,
                                         key=f"rate_{ref_id}_{trial_id}")
        rating = rating_dict[rating_choice]
    elif experiment.rating_system == '5 star system':
        rating = st.slider('How many stars would you rate this out of 5?',
                           min_value=1.,
                           max_value=5.,
                           step=0.25,
                           key=f"rate_{ref_id}_{trial_id}")
    elif experiment.rating_system == 'percent system':
        rating = st.number_input('How would you rate this from (0-100%)?',
                                 min_value=0,
                                 max_value=100,
                                 step=1,
                                 key=f"rate_{ref_id}_{trial_id}")
    elif experiment.rating_system == 'unbounded system':
        rating = st.number_input('How would you rate this (you can choose any number)?',
                                 step=0.01,
                                 key=f"rate_{ref_id}_{trial_id}")
    else:
        st.error("Unable to record rating. Please contact whoever gave your this rating code.")
        return
    if st.button("Submit rating", key=f"submit_rating_{ref_id}_{trial_id}"):
        # Get experiment, and store trial update
        bo_experiment = BayesianOptimisation(experiment=experiment.opt_experiment)
        bo_experiment.post_measurement(
            trial_id=trial_id,
            trial_update=TrialUpdate(ref_id=ref_id, objective_measurement=rating)
        )
        store_experiment(experiment=experiment)
        st.info(f"Thank you for your rating!")


def rate_batch(access_code: str):
    feedback_map = get_feedback_map()
    if access_code not in feedback_map.feedback_refs:
        st.info(f"Invalid code {access_code}.")
        return
    feedback_ref = feedback_map.feedback_refs[access_code]
    experiment = get_experiment(experiment_id=feedback_ref.experiment_id)
    if experiment is None:
        st.info("Unable to leave feedback. Code 001")
        return
    if feedback_ref.trial_id not in experiment.batch_meta:
        st.info('Unable to leave feedback. Code 002')
        return
    batch_meta = experiment.batch_meta[feedback_ref.trial_id]
    st.header("Your feedback is highly appreciated!")
    if batch_meta.image is not None:
        st.image(str_to_bytes(batch_meta.image), caption=batch_meta.name)
    ask_rating(ref_id=access_code, trial_id=feedback_ref.trial_id,
               experiment=experiment)
