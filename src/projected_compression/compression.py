from collections import OrderedDict
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectedLinear(nn.Module):
    __constants__ = ["in_features", "out_features", "base_in_features", "base_out_features"]
    base_in_features: int
    base_out_features: int
    projected_in_features: Optional[int]
    projected_out_features: Optional[int]


    def __init__(
        self,
        base_in_features: int,
        base_out_features: int,
        projected_in_features: Optional[int],
        projected_out_features: Optional[int],
        device=None,
        dtype=None,
    ) -> nn.Module:

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.base_in_features = base_in_features
        self.base_out_features = base_out_features
        self.projected_in_features = projected_in_features
        self.projected_out_features = projected_out_features
        self.finalized_projections = False

        if self.projected_in_features is not None: 
            self.projection_in_weight = nn.Parameter(
                torch.zeros((projected_in_features, base_in_features), **factory_kwargs)
            )

        self.base_weight = nn.Parameter(
            torch.rand((base_in_features, base_out_features), **factory_kwargs)
        )

        if self.projected_out_features is not None:
            self.projection_out_weight = nn.Parameter(
                torch.zeros((base_out_features, projected_out_features), **factory_kwargs)
            )    

        self.final_weight = nn.Parameter(
            torch.empty((projected_out_features, projected_in_features), **factory_kwargs)
        )
    
    def init_projections(self):
        with torch.no_grad():
            if self.projected_in_features is not None: 
                l2_norms = torch.norm(self.base_weight, dim=1)
                top_indices = torch.topk(l2_norms, self.projected_in_features).indices
                self.projection_in_weight[torch.arange(self.projected_in_features), top_indices] = 1

            if self.projected_out_features is not None: 
                l2_norms = torch.norm(self.base_weight, dim=0)
                top_indices = torch.topk(l2_norms, self.projected_out_features).indices
                self.projection_out_weight[top_indices, torch.arange(self.projected_out_features)] = 1


    def finalize_projections(self):
        projected_base_weight = self.base_weight

        if self.projected_in_features is not None:
            projected_base_weight = self.projection_in_weight @ self.base_weight

        if self.projected_out_features is not None:
            projected_base_weight = projected_base_weight @ self.projection_out_weight

        self.final_weight.data.copy_(projected_base_weight.T)
        self.finalized_projections = True


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # gradient magic happens here - PC optimization
        # TODO maybe order of projections matter for speed
        if self.finalized_projections:
            return F.linear(input, self.final_weight, bias = None)

        projected_base_weight = self.base_weight

        if self.projected_in_features is not None:
            projected_base_weight = self.projection_in_weight @ self.base_weight

        if self.projected_out_features is not None:
            projected_base_weight = projected_base_weight @ self.projection_out_weight

        return F.linear(input, projected_base_weight.T, bias = None)
        

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, base_in_features = {self.__project_in}, {self.base_in_features}, base_out_features = {self.__project_out}, {self.base_out_features}" # no bias
    

    def state_dict(self, *args, **kwargs):
        full_state = super().state_dict(*args, **kwargs)
        if self.finalized_projections:
            full_state = super().state_dict(*args, **kwargs)
            return {"final_weight": full_state["final_weight"]}

        return full_state
    
    # def _training_key_params(self) -> set:
    #     training_params = {"base_weight"}
    #     if self.projected_in_features is not None:
    #         training_params.add("projection_in_weight")
        
    #     if self.projected_out_features is not None:
    #         training_params.add("projection_out_weight")

    def load_state_dict(self, state_dict, strict: bool = True, assign=False):
        if self.finalized_projections:
            if {"final_weight"} == state_dict.keys():
                self.final_weight.data.copy_(state_dict["final_weight"])
            elif strict:
                raise RuntimeError("Missing 'final_weight' in state_dict")
        
        else:
            super().load_state_dict(state_dict, assign)
            

        # training_key_params = self._training_key_params()
        # if strict:
        #     assert state_dict.keys() ==  training_key_params, "LinearProjection state keys does not match!"

        # for key in training_key_params:
        #     self[key].data.copy_(state_dict[key])


# TODO remove this TEST
input_data = torch.rand((6,2))

proj_linear = ProjectedLinear(6, 8, 2, 4)
proj_linear.init_projections()

output1 = proj_linear(input_data)
print(output1)

state = proj_linear.state_dict()
print(state)

proj_linear.load_state_dict(state)

proj_linear.finalize_projections()

output2 = proj_linear(input_data)
print(output2)

print(output1 == output2)

state = proj_linear.state_dict()
print(state)

proj_linear.load_state_dict(state)