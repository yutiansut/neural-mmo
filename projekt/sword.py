from pdb import set_trace as T
from collections import defaultdict
import numpy as np 
import ray
import projekt

from forge import trinity

from forge.blade.core import realm
from forge.blade.lib.log import Blob, BlobSummary

from forge.ethyr.io import Stimulus, Action
from forge.ethyr.experience import RolloutManager
from forge.ethyr.torch import Model, optim
from forge.ethyr.torch.param import setParameters, getParameters

from forge.ethyr.io.io import IO, Output

from copy import deepcopy

from forge.trinity.ascend import Ascend, runtime

#Currently, agents technically run on the same core
#as the environment. This saves 2x cores at small scale
#but will not work with a large number of agents.
#Enable @ray.remote when this becomes an issue.
#@ray.remote
class Sword(Ascend):
   '''Core level Sword API demo

   This core level client node maintains a
   full copy of the model. It runs and computes
   updates for the associated policies of all
   agents.'''

   def __init__(self, trinity, config, idx):
      '''Initializes a model and relevent utilities'''
      super().__init__(None, 0)
      config        = deepcopy(config)
      self.config   = config
      self.ent      = 0

      self.keys = set()

      self.params   = None 

      self.reward   = {}
      self.noise    = {}
      self.net      = {}
      self.blobs    = {}
      self.summary  = BlobSummary()
   
   @runtime
   def step(self, obs, packet=None):
      '''Synchronizes weights from upstream; computes
      agent decisions; computes policy updates.
      
      A few bug notes:
         1. It appears pytorch errors in .backward when batching
         data. This is because the graph is retained over all
         trajectories in the batch, even though only some are
         finished.
         
         2. Currently specifying retain_graph. This should not be
         required with batch size 1, even with the above bug.
      '''
      actions = {}
      n       = 96833
      config  = self.config

      #Update base parmeters
      if packet is not None:
         self.params = packet
      
      for ob in obs:
         iden = ob.entID

         #Add new network
         if iden not in self.net:
            noise = config.ES_STD * np.random.randn(n)

            self.noise[iden] = noise
            self.net[iden]   = projekt.ANN(config)
            self.blobs[iden] = Blob(ob.entID, ob.annID)

            setParameters(self.net[iden], self.params + noise)
         self.blobs[iden].update()

         #Finish old networks
         if ob.done:
            self.reward[iden] = ob.reward 

            self.summary.nRollouts += 1
            self.summary.nUpdates  += ob.reward
            self.summary.blobs.append(self.blobs[iden])

            del self.net[iden]
            del self.blobs[iden]

         #Compute forward pass
         else:
            key = ob.key
            #Pseudo batch single observation
            stim, atn = IO.batch([ob])

            #Compute forward pass
            atn, atnIdx, val = self.net[ob.entID](0, stim, atn)

            #Collect output actions and values
            #atns are detached in torch/io/action
            atn    = atn[0]
            atnIdx = atnIdx.detach()[0]
            val    = val.detach()[0]

            out = Output(key, atn, atnIdx, val)
            actions.update(out.action)
            
      #Collect updates
      summary = self.summary
      if summary.nUpdates >= config.CLIENT_UPDATES:
         rewards = list(self.reward.values())
         base_mean = np.mean(rewards)
         base_std  = np.std(rewards)
         self.summary = BlobSummary()
         update = 0

         if config.TEST:
            return actions, None, summary  

         keys = list(self.reward.keys())
         for iden in keys:
            reward = self.reward[iden]
            noise  = self.noise[iden]

            reward = (reward - base_mean) / base_std
            update += reward * noise / summary.nRollouts

            del self.noise[iden]
            del self.reward[iden]

         return actions, update, summary

      return actions, None, None

