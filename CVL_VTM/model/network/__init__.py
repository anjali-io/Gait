# model/network/__init__.py

from .triplet import TripletLoss

# CVL-aware model (MAIN ENTRY)
from .cvl_model import CVL_GaitSet

# CVL components
from .vtm import ViewTransitionModule
from .cvl_loss import CVLLoss



