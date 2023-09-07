import base64
import inspect
import logging
from datetime import datetime
from typing import List, Type, TypeVar, Dict, Any
from uuid import uuid4

import aiohttp
import streamlit as st
from pydantic import BaseModel, Field, ValidationError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ray')


class APIError(Exception):
    pass


def example_from_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    """
    Generate example from schema and return as dict.

    Args:
        model: BaseModel

    Returns: dict of example
    """
    example = dict()
    properties = model.schema().get('properties', dict())
    for field in model.__fields__:
        # print(model, model.__fields__[field])
        if inspect.isclass(model.__fields__[field]):
            if issubclass(model.__fields__[field], BaseModel):
                example[field] = example_from_schema(model.__fields__[field])
                continue
            example[field] = None
        example[field] = properties[field].get('example', None)
        # print(field, example[field])
    return example


_T = TypeVar('_T')


def build_example(model: Type[_T]) -> _T:
    """
    Build example for model from it's schema and `example` field properties.

    Args:
        model: BaseModel

    Returns:
        example of model
    """
    return model(**example_from_schema(model))


class ParameterResponse(BaseModel):
    id: str = Field(
        description="The id of the parameter. This is the clients reference ID.",
        example=str(uuid4())
    )
    name: str = Field(
        description="The name of the parameter.",
        example="price"
    )
    current_trial_id: str = Field(
        description="The id of the trial that this parameter is currently set to.",
        example=str(uuid4())
    )
    current_value: float | int = Field(
        description="The value of the parameter that is currently set.",
        example=0.99
    )
    updated_at: datetime = Field(
        description="The time that the parameter was last updated. With tzinfo set.",
        example=datetime.fromisoformat(datetime.now().isoformat() + "Z")
    )
    is_active: bool = Field(
        description="Whether this parameter is currently active.",
        example=True
    )


class ObservableResponse(BaseModel):
    observable_name: str = Field(
        description="A unique id of the parameter that this observation is for.",
        example='is_subscribed'
    )
    observable: float | int = Field(
        description="The value of the observable.",
        example=1
    )


class UserObservableResponse(BaseModel):
    user_id: str = Field(
        description="A unique id of the user that this observation is for.",
        example=str(uuid4())
    )
    trial_id: str = Field(
        description="The id of the trial that this observation is for.",
        example=str(uuid4())
    )
    observables: List[ObservableResponse] = Field(
        description="A list of observables for this user.",
        example=[
            build_example(ObservableResponse)
        ]
    )


class PullParametersResponse(BaseModel):
    current_parameters: List[ParameterResponse] = Field(
        description="A list of the current parameters.",
        example=[
            build_example(ParameterResponse)
        ]
    )


class PullObservablesResponse(BaseModel):
    user_observations: List[UserObservableResponse] = Field(
        description="A list of the user observations.",
        example=[
            build_example(UserObservableResponse)
        ]
    )


class ParameterRequest(BaseModel):
    parameter_name: str = Field(
        description="The name of the parameter.",
        example="price"
    )
    value: float | int = Field(
        description="The value of the parameter.",
        example=0.99
    )


class PushRequest(BaseModel):
    trial_id: str = Field(
        description="The id of the trial that this parameter is currently set to.",
        example=str(uuid4())
    )
    parameters: List[ParameterRequest] = Field(
        description="A list of the parameters to push.",
        example=[
            build_example(ParameterRequest)
        ]
    )


class NoTrialSet(Exception):
    pass


def test_examples():
    # print(build_example(PushRequest).schema_json(indent=2))
    # print(build_example(PushRequest).json(indent=2))
    # print(build_example(ParameterResponse).schema_json(indent=2))
    # print(build_example(ParameterResponse).json(indent=2))
    # print(build_example(UserObservableResponse).schema_json(indent=2))
    # print(build_example(UserObservableResponse).json(indent=2))
    pass


class ABInterface:
    """
    Represents an exposed interface to a client running AB test. 
    Authentication is bearer auth.
    """

    def __init__(self, client_api_url: str, client_api_key: str):
        self.backend_url = client_api_url
        # Decode key
        key_hash = client_api_key
        key_hash_bytes = base64.b64decode(key_hash)
        decoded_key = key_hash_bytes.decode('ascii')

        self.be_headers = {'Authorization': f'Bearer {decoded_key}',
                           'Content-Type': 'application/json'}

    async def push_new_trial(self, push_request: PushRequest):
        """
        Pushes a new trial to client with POST /ab/parameters/new-trial
        
        Args:
            push_request: PushRequest object
            
        Raises:
            RuntimeError if 201 not received.
        """
        async with aiohttp.ClientSession(headers=self.be_headers) as session:
            async with session.post(f'{self.backend_url}/ab/parameters/new-trial',
                                    data=push_request.json()) as response:
                status = response.status
                if status != 201:
                    raise APIError(f"Failed to create trial {await response.text()} Code {status}.")

    async def pull_parameter_data(self) -> PullParametersResponse:
        """
        Gets the current parameters set by client (GET /ab/parameters).

        Returns:
            a PullParametersResponse object
        """
        async with aiohttp.ClientSession(headers=self.be_headers) as session:
            async with session.get(f'{self.backend_url}/ab/parameters') as response:
                status = response.status
                if status != 200:
                    raise APIError(f"Failed to get current parameters {await response.text()} Code {status}.")
                else:
                    resp_json = await response.json()
                    current_parameters = []
                    for _resp in resp_json:
                        try:
                            current_parameters.append(ParameterResponse.parse_obj(_resp))
                        except ValidationError as e:
                            raise NoTrialSet(f"No trial set for {_resp}. {str(e)}")
        return PullParametersResponse(
            current_parameters=current_parameters
        )

    async def pull_observable_data(self) -> PullObservablesResponse:
        """
        Gets the current observables set by client (GET /ab/observations).

        Returns:
            a PullObservablesResponse object
        """
        async with aiohttp.ClientSession(headers=self.be_headers) as session:

            async with session.get(f'{self.backend_url}/ab/observations') as response:
                status = response.status
                if status != 200:
                    raise APIError(f"Failed to get observations {await response.text()} Code {status}.")
                else:
                    user_observations: List[UserObservableResponse] = []
                    resp_json = await response.json()
                    for _resp in resp_json:
                        user_observations.append(UserObservableResponse.parse_obj(_resp))
        return PullObservablesResponse(
            user_observations=user_observations
        )
