import base64
import logging
from datetime import datetime
from typing import List

import aiohttp
from pydantic import ValidationError, BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ray')


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
                    raise RuntimeError(f"Failed to create trial {await response.text()} Code {status}.")

    async def pull_observable_data(self) -> PullResponse:
        """
        Gets the current parameters set by client (GET /ab/parameters), and the list of observations (GET /ab/observations).
        
        Returns:
            a PullResponse object 
        """
        async with aiohttp.ClientSession(headers=self.be_headers) as session:
            async with session.get(f'{self.backend_url}/ab/parameters') as response:
                status = response.status
                if status != 200:
                    raise RuntimeError(f"Failed to get current parameters {await response.text()} Code {status}.")
                else:
                    resp_json = await response.json()
                    current_parameters = []
                    for _resp in resp_json:
                        try:
                            current_parameters.append(ParameterResponse.parse_obj(_resp))
                        except ValidationError:
                            raise NoTrialSet(name=_resp['name'])

            async with session.get(f'{self.backend_url}/ab/observations') as response:
                status = response.status
                if status != 200:
                    logger.info(f"Failed to get observations {await response.text()} Code {status}.")
                else:
                    user_observations: List[UserObservableResponse] = []
                    for inner_array in (await response.json()):
                        user_observations.extend(map(UserObservableResponse.parse_obj, inner_array))
        return PullResponse(
            current_parameters=current_parameters,
            user_observations=user_observations
        )
