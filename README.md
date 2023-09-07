# YourBayesOptimiser

Your Bayesian Optimiser App

## Integration

For integrating with your own service you need to expose three methods endpoints:

## `POST /ab/parameters/new-trial`

Sets a new trial in the client. A trial corresponds to a particular set of parameters.
It is expected that this trial-id will be provided back when observables are fetched from the client later.
For example, if the client has users, then new users will be given the set of parameters that are currently set in the
client, and link that user with the trial. Observables from this user will provide the trial id as well.

Takes a `PushRequest` as data.

<details open>
  <summary>Schema</summary>

```json
{
  "title": "PushRequest",
  "type": "object",
  "properties": {
    "trial_id": {
      "title": "Trial Id",
      "description": "The id of the trial that this parameter is currently set to.",
      "example": "77b9e20a-90f2-4706-8d0c-2b6064ce3b6c",
      "type": "string"
    },
    "parameters": {
      "title": "Parameters",
      "description": "A list of the parameters to push.",
      "example": [
        {
          "parameter_name": "price",
          "value": 0.99
        }
      ],
      "type": "array",
      "items": {
        "$ref": "#/definitions/ParameterRequest"
      }
    }
  },
  "required": [
    "trial_id",
    "parameters"
  ],
  "definitions": {
    "ParameterRequest": {
      "title": "ParameterRequest",
      "type": "object",
      "properties": {
        "parameter_name": {
          "title": "Parameter Name",
          "description": "The name of the parameter.",
          "example": "price",
          "type": "string"
        },
        "value": {
          "title": "Value",
          "description": "The value of the parameter.",
          "example": 0.99,
          "anyOf": [
            {
              "type": "number"
            },
            {
              "type": "integer"
            }
          ]
        }
      },
      "required": [
        "parameter_name",
        "value"
      ]
    }
  }
}
```

</details>


<details open>
  <summary>Example</summary>

This is an example of a payload, that sets a single parameter.

```json
{
  "trial_id": "77b9e20a-90f2-4706-8d0c-2b6064ce3b6c",
  "parameters": [
    {
      "parameter_name": "price",
      "value": 0.99
    }
  ]
}
```

</details>

## `GET /ab/parameters`

Returns the currently set parameters in the client, as an array of `ParameterResponse`.
Since each parameter may be set from a different trial, the trial id is also returned per parameter.
E.g. you may have two experiments on going adjusting two separate parameters. **Warning**: having more than one
experiment on-going is risky as it does not control for interaction between the experiments. This is up to the client to
take into consideration.

<details open>
  <summary>Schema</summary>

```json
{
  "title": "ParameterResponse",
  "type": "object",
  "properties": {
    "id": {
      "title": "Id",
      "description": "The id of the parameter. This is the clients reference ID.",
      "example": "69b57c42-f530-43c5-8252-294c6b9de3f3",
      "type": "string"
    },
    "name": {
      "title": "Name",
      "description": "The name of the parameter.",
      "example": "price",
      "type": "string"
    },
    "current_trial_id": {
      "title": "Current Trial Id",
      "description": "The id of the trial that this parameter is currently set to.",
      "example": "0a33998e-97d8-45ab-9a7a-3f3bbefeb6f4",
      "type": "string"
    },
    "current_value": {
      "title": "Current Value",
      "description": "The value of the parameter that is currently set.",
      "example": 0.99,
      "anyOf": [
        {
          "type": "number"
        },
        {
          "type": "integer"
        }
      ]
    },
    "updated_at": {
      "title": "Updated At",
      "description": "The time that the parameter was last updated. With tzinfo set.",
      "example": "2023-09-07T15:27:19.779760+00:00",
      "type": "string",
      "format": "date-time"
    },
    "is_active": {
      "title": "Is Active",
      "description": "Whether this parameter is currently active.",
      "example": true,
      "type": "boolean"
    }
  },
  "required": [
    "id",
    "name",
    "current_trial_id",
    "current_value",
    "updated_at",
    "is_active"
  ]
}
```

</details>


<details open>
  <summary>Example</summary>

This is an example of a response from the server, with a single element in array.

```json
[
  {
    "id": "69b57c42-f530-43c5-8252-294c6b9de3f3",
    "name": "price",
    "current_trial_id": "0a33998e-97d8-45ab-9a7a-3f3bbefeb6f4",
    "current_value": 0.99,
    "updated_at": "2023-09-07T15:27:19.779760+00:00",
    "is_active": true
  }
]
```

</details>

## `GET /ab/observations`

Returns all the observables from the client in an array. If your client has users then typically you would be measuring
something from users, like whether they subscribed, and you would return that for each user, along with the trial each
user is part of. These observables are used to calculate the trial scores, using python expressions.

Returns an array of `UserObservableResponse` as data.

<details open>
  <summary>Schema</summary>

```json
{
  "title": "UserObservableResponse",
  "type": "object",
  "properties": {
    "user_id": {
      "title": "User Id",
      "description": "A unique id of the user that this observation is for.",
      "example": "6b299dd1-d3cf-48a1-9762-dd35fb98c237",
      "type": "string"
    },
    "trial_id": {
      "title": "Trial Id",
      "description": "The id of the trial that this observation is for.",
      "example": "30c6648d-2d50-4759-a44b-d9640642cd3d",
      "type": "string"
    },
    "observables": {
      "title": "Observables",
      "description": "A list of observables for this user.",
      "example": [
        {
          "observable_name": "is_subscribed",
          "observable": 1.0
        }
      ],
      "type": "array",
      "items": {
        "$ref": "#/definitions/ObservableResponse"
      }
    }
  },
  "required": [
    "user_id",
    "trial_id",
    "observables"
  ],
  "definitions": {
    "ObservableResponse": {
      "title": "ObservableResponse",
      "type": "object",
      "properties": {
        "observable_name": {
          "title": "Parameter Id",
          "description": "A unique id of the parameter that this observation is for.",
          "example": "is_subscribed",
          "type": "string"
        },
        "observable": {
          "title": "Observable",
          "description": "The value of the observable.",
          "example": 1,
          "anyOf": [
            {
              "type": "number"
            },
            {
              "type": "integer"
            }
          ]
        }
      },
      "required": [
        "observable_name",
        "observable"
      ]
    }
  }
}
```

</details>


<details open>
  <summary>Example</summary>

This is an example of a response from the server, with a single element in array.

```json
[
  {
    "user_id": "6b299dd1-d3cf-48a1-9762-dd35fb98c237",
    "trial_id": "30c6648d-2d50-4759-a44b-d9640642cd3d",
    "observables": [
      {
        "observable_name": "is_subscribed",
        "observable": 1.0
      }
    ]
  }
]
```

</details>
