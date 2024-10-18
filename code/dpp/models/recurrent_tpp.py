import dpp
import pandas as pd
import torch
import torch.nn as nn
import cProfile
import time

from datetime import datetime
from torch.distributions import Categorical

from dpp.data.batch import Batch
from dpp.utils import diff
from typing import List
import torch.nn.functional as F


class RecurrentTPP(nn.Module):
    """
    RNN-based TPP model for marked and unmarked event sequences.

    The marks are assumed to be conditionally independent of the inter-event times.

    Args:
        num_marks: Number of marks (i.e. classes / event types)
        num_meta_classes: Number of mmeta classes
        mean_log_inter_time: Average log-inter-event-time, see dpp.data.dataset.get_inter_time_statistics
        std_log_inter_time: Std of log-inter-event-times, see dpp.data.dataset.get_inter_time_statistics
        context_size: Size of the context embedding (history embedding)
        mark_embedding_size: Size of the mark embedding (used as RNN input)
        rnn_type: Which RNN to use, possible choices {"RNN", "GRU", "LSTM"}

    """
    def __init__(
        self,
        use_src_marks: bool,
        use_dst_marks: bool,
        num_src_marks: int,
        num_dst_marks: int,
        num_meta_classes: int,
        meta_type: "basic",
        mean_log_inter_time: float = 0.0,
        std_log_inter_time: float = 1.0,
        context_size: int = 32,
        src_mark_embedding_size: int = 32,
        dst_mark_embedding_size: int = 32,
        shared_mark_embedding: bool = False,
        meta_embedding_size: int = 32,
        rnn_type: str = "GRU",
    ):
        super().__init__()
        self.use_src_marks = use_src_marks
        self.use_dst_marks = use_dst_marks
        self.num_src_marks = num_src_marks
        self.num_dst_marks = num_dst_marks
        self.num_meta_classes = num_meta_classes
        self.meta_type = meta_type
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        self.context_size = context_size
        self.src_mark_embedding_size = src_mark_embedding_size
        self.dst_mark_embedding_size = dst_mark_embedding_size
        self.shared_mark_embedding = shared_mark_embedding
        self.meta_embedding_size = meta_embedding_size
        self.num_features = 1
        
        ###### NEW MARK LOGIC
        if self.use_src_marks:
            ## We have sender marks
            self.src_mark_embedding = nn.Embedding(self.num_src_marks, self.src_mark_embedding_size)
            src_linear_units=150
            self.src_mark_linear1 = nn.Linear(self.context_size, src_linear_units)
            self.src_mark_linear2 = nn.Linear(src_linear_units, self.num_src_marks)
            self.num_features += self.src_mark_embedding_size
            if self.use_dst_marks:
                ## We also have destination marks
                if self.shared_mark_embedding:
                    if self.src_mark_embedding_size != self.dst_mark_embedding_size:
                        raise ValueError(f"Shared embedding specified but embedding sizes are different (got src={src_mark_embedding_size}, dst={dst_mark_embedding_size})")
                    self.dst_mark_embedding = self.src_mark_embedding
                else:
                    self.dst_mark_embedding = nn.Embedding(self.num_dst_marks, self.dst_mark_embedding_size)
                dst_linear_units = 250
                self.dst_mark_linear = nn.Linear(self.context_size + self.src_mark_embedding_size, self.num_dst_marks)
                self.num_features += self.dst_mark_embedding_size
        
        if self.num_meta_classes > 0:
            self.meta_embedding = nn.Embedding(self.num_meta_classes, self.meta_embedding_size)    
        self.rnn_type = rnn_type
        self.context_init = nn.Parameter(torch.zeros(context_size))  # initial state of the RNN
        self.on_fly = None
        self.rnn = getattr(nn, rnn_type)(input_size=self.num_features, hidden_size=self.context_size, batch_first=True)

    def get_features(self, batch: dpp.data.Batch) -> torch.Tensor:
        """
        Convert each event in a sequence into a feature vector.

        Args:
            batch: Batch of sequences in padded format (see dpp.data.batch).

        Returns:
            features: Feature vector corresponding to each event,
                shape (batch_size, seq_len, num_features)

        """
        features = torch.log(batch.inter_times + 1e-8).unsqueeze(-1)  # (batch_size, seq_len, 1)
        features = (features - self.mean_log_inter_time) / self.std_log_inter_time
        if self.use_src_marks:
            src_mark_emb = self.src_mark_embedding(batch.src_marks)  # (batch_size, seq_len, mark_embedding_size)
            features = torch.cat([features, src_mark_emb], dim=-1)
            if self.use_dst_marks:
                dst_mark_emb = self.dst_mark_embedding(batch.dst_marks)
                features = torch.cat([features, dst_mark_emb], dim=-1)
                
        return features  # (batch_size, seq_len, num_features)

    def get_context(self, features: torch.Tensor, remove_last: bool = True) -> torch.Tensor:
        """
        Get the context (history) embedding from the sequence of events.

        Args:
            features: Feature vector corresponding to each event,
                shape (batch_size, seq_len, num_features)
            remove_last: Whether to remove the context embedding for the last event.

        Returns:
            context: Context vector used to condition the distribution of each event,
                shape (batch_size, seq_len, context_size) if remove_last == False
                shape (batch_size, seq_len + 1, context_size) if remove_last == True

        """
        context = self.rnn(features)[0]
        batch_size, seq_len, context_size = context.shape
        context_init = self.context_init[None, None, :].expand(batch_size, 1, -1)  # (batch_size, 1, context_size)
        # Shift the context by vectors by 1: context embedding after event i is used to predict event i + 1
        if remove_last:
            context = context[:, :-1, :]
        context = torch.cat([context_init, context], dim=1)
        return context

    def get_inter_time_dist(self, context: torch.Tensor) -> torch.distributions.Distribution:
        """
        Get the distribution over inter-event times given the context.

        Args:
            context: Context vector used to condition the distribution of each event,
                shape (batch_size, seq_len, context_size)

        Returns:
            dist: Distribution over inter-event times, has batch_shape (batch_sizesampled_batch, seq_len)

        """
        raise NotImplementedError()

    def log_prob(self, batch: dpp.data.Batch) -> torch.Tensor:
        """Compute log-likelihood for a batch of sequences.

        Args:
            batch:

        Returns:
            log_p: shape (batch_size,)

        """            
        time_scale_factor = 1
        src_scale_factor = 1
        dst_scale_factor = 1
        features = self.get_features(batch)
        context = self.get_context(features)
        if self.num_meta_classes > 0:
            meta_emb = self.meta_embedding(batch.meta)
            context_meta = torch.cat([context, meta_emb], dim=-1)
        inter_time_dist = self.get_inter_time_dist(context_meta)
        
        inter_times = batch.inter_times.clamp(1e-10)
        time_log_p = inter_time_dist.log_prob(inter_times)  # (batch_size, seq_len)
        log_p = time_log_p/time_scale_factor 
        # Survival probability of the last interval (from t_N to t_end).
        last_event_idx = batch.mask.sum(-1, keepdim=True).long()  # (batch_size, 1)
        log_surv_all = inter_time_dist.log_survival_function(inter_times)  # (batch_size, seq_len)
        log_surv_last = torch.gather(log_surv_all, dim=-1, index=last_event_idx).squeeze(-1)  # (batch_size,)

        if self.use_src_marks:
            src_context = self.src_mark_linear1(context)
            lin1src_context = F.tanh(src_context)
            lin2src_context = self.src_mark_linear2(lin1src_context)
            src_mark_logits = torch.log_softmax(lin2src_context, dim=-1)
            sender_accuracy = (batch.src_marks == src_mark_logits.argmax(-1)).float()
            src_mark_dist = Categorical(logits=src_mark_logits)
            src_log_p = src_mark_dist.log_prob(batch.src_marks)
            log_p += src_log_p/src_scale_factor
            topk_sender = self.accuracy(src_mark_logits, batch.src_marks, batch.mask, topk=[3])
            if self.use_dst_marks:
                dst_mark_logits = torch.log_softmax(self.dst_mark_linear(torch.cat([context, self.src_mark_embedding(batch.src_marks)], dim=-1)), dim=-1)  
                recip_accuracy = (batch.dst_marks == torch.squeeze(dst_mark_logits.argmax(-1))).float()
                topk_recip = self.accuracy(dst_mark_logits, batch.dst_marks, batch.mask, topk=[3])
                dst_mark_dist = Categorical(logits=dst_mark_logits)
                dst_log_p = dst_mark_dist.log_prob(batch.dst_marks)
                log_p += dst_log_p/dst_scale_factor
                
        log_p *= batch.mask  # (batch_size, seq_len)
        if self.use_dst_marks:
            return log_p.sum(-1) + 1/time_scale_factor*log_surv_last, 1/time_scale_factor*((batch.mask*time_log_p).sum(-1) + log_surv_last), (batch.mask*src_log_p).sum(-1), (batch.mask*dst_log_p).sum(-1), batch.mask*sender_accuracy, topk_sender, batch.mask*recip_accuracy, topk_recip  
        else: 
            return log_p.sum(-1) + 1/time_scale_factor*log_surv_last, 1/time_scale_factor*((batch.mask*time_log_p).sum(-1) + log_surv_last), (batch.mask*src_log_p).sum(-1), batch.mask*sender_accuracy, topk_sender 
    
    def accuracy(self, output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, topk=(1,)) -> List[torch.FloatTensor]:
        """
        Computes the accuracy over the k top predictions for the specified values of k
        In top-5 accuracy you give yourself credit for having the right answer
        if the right answer appears in your top five guesses.

        ref:
        - https://pytorch.org/docs/stable/generated/torch.topk.html
        - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
        - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
        - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
        - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

        :param output: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
        :param target: target is the truth
        :param topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
        e.g. in top 2 it means you get a +1 if your models's top 2 predictions are in the right label.
        So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
        but if it were either cat or dog you'd accumulate +1 for that example.
        :return: list of topk accuracy [top1st, top2nd, ...] depending on your topk input
        """
        with torch.no_grad():
            # ---- get the topk most likely labels according to your model
            # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
            maxk = max(topk)  # max number labels we will consider in the right choices for out model
            batch_size = target.size(0)

            # get top maxk indicies that correspond to the most likely probability scores
            # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
            _, y_pred = output.topk(k=maxk, dim=-1)  # _, [B, n_classes] -> [B, maxk]
            #print(y_pred.shape, y_pred.view(-1, maxk).shape)#, target.unsqueeze(-1).shape, mask.shape)
            y_pred = y_pred.view(-1, maxk).t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

            # - get the credit for each example if the models predictions is in maxk values (main crux of code)
            # for any example, the model will get credit if it's prediction matches the ground truth
            # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
            # if the k'th top answer of the model matches the truth we get 1.
            # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
            target_reshaped = target.flatten().view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
            # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
            
            mask_reshaped = mask.flatten().view(1, -1).expand_as(y_pred)
            correct = (y_pred == target_reshaped)*mask_reshaped # [maxk, B] were for each example we know which topk prediction matched truth


        return correct  # list of topk accuracies for entire batch [topk1, topk2, ... etc]
    
    
    def mae(self, batch: dpp.data.Batch) -> torch.Tensor:
        """Compute log-likelihood for a batch of sequences.

        Args:
            batch:

        Returns:
            log_p: shape (batch_size,)

        """
        features = self.get_features(batch)
        context = self.get_context(features)
        if self.num_meta_classes > 0:
            meta_emb = self.meta_embedding(batch.meta)
            context_meta = torch.cat([context, meta_emb], dim=-1)
        inter_time_dist = self.get_inter_time_dist(context_meta)
        
        inter_times = batch.inter_times.clamp(1e-10)
        mean_time = inter_time_dist.mean
        mae = abs(mean_time - inter_times)  # (batch_size, seq_len)
        rmse = ((mean_time - inter_times) ** 2)

        return mae, rmse

    def sample(self, t_start: float, t_end: float, batch_size: int = 1, context_init: torch.Tensor = None) -> dpp.data.Batch:
        """Generate a batch of sequence from the model.

        Args:
            t_end: Size of the interval on which to simulate the TPP.
            batch_size: Number of independent sequences to simulate.
            context_init: Context vector for the first event.
                Can be used to condition the generator on past events,
                shape (context_size,)

        Returns;
            batch: Batch of sampled sequences. See dpp.data.batch.Batch.
        """
        
        def compute_meta(inter_times, meta_type='basic'):
            if len(inter_times) > 0:
                ts = t_start + inter_times.sum(-1).item()
                # 1848 of hours of sequences were thrown away at the start of the Enron dataset
                date = datetime(1999, 3, 4, 9, 39).astimezone() + pd.to_timedelta(ts + 1848,'h')
                hour = int(date.strftime("%H"))
                weekday = date.weekday()
                week = int(date.strftime("%V"))
                year = int(date.strftime("%Y")) - 1999
                if meta_type == '247':
                    meta = torch.tensor([[7 * weekday + hour]])
                elif meta_type == 'tempo':
                    meta = torch.tensor([[get_tempo_meta(week, year)]])
                elif meta_type == 'basic':    
                    meta = torch.tensor([[get_meta(hour, weekday)]])
            else:
                meta = torch.tensor([[0]])
            return meta

        def get_meta(hour, weekday):
            if weekday in range(7):
                # It's a weekday
                if hour in range(10, 16):
                    # Regular office hours
                    return 0
                elif hour in [7, 8, 9, 16, 17, 18, 19, 20]:
                    # Shoulder
                    return 1
                else:
                    # Non-work hours
                    return 2
            else:
                # It's the weekend
                return 3
    
        def get_tempo_meta(week, year):
            return year * 52 + week - 9
        
        if context_init is None:
            # Use the default context vector
            context_init = self.context_init
        else:
            # Use the provided context vector
            context_init = context_init.view(self.context_size)
        next_context = context_init[None, None, :].expand(batch_size, 1, -1)

        inter_times = torch.empty(batch_size, 0)
        if self.use_src_marks:
            src_marks = torch.empty(batch_size, 0, dtype=torch.long)
            if self.use_dst_marks:
                dst_marks = torch.empty(batch_size, 0, dtype=torch.long)
                
        if self.num_meta_classes > 0:
            metas = torch.empty(batch_size, 0, dtype=torch.long)    
            meta_emb = self.meta_embedding(torch.ones(batch_size, 1, dtype=torch.long))
            next_meta_context = torch.cat([next_context, meta_emb], dim=-1)
            
        generated = False
        i = 0
        while not generated and inter_times.sum(-1).min() < t_end:
            inter_time_dist = self.get_inter_time_dist(next_meta_context)
            next_inter_times = inter_time_dist.sample()  # (batch_size, 1)
            inter_times = torch.cat([inter_times, next_inter_times], dim=1)  # (batch_size, seq_len)
            i += 1
            if i%1000 == 0:
                print(i, inter_times.sum(-1).min())
            
            # Generate marks, if necessary
            if self.use_src_marks:
                src_context = self.src_mark_linear1(next_context)
                lin1src_context = F.tanh(src_context)
                lin2src_context = self.src_mark_linear2(lin1src_context)
                src_mark_logits = torch.log_softmax(lin2src_context, dim=-1)
                src_mark_dist = Categorical(logits=src_mark_logits)
                next_src_marks = src_mark_dist.sample()  # (batch_size, 1)
                src_marks = torch.cat([src_marks, next_src_marks], dim=1)
                if self.use_dst_marks:
                    dst_mark_logits = torch.log_softmax(self.dst_mark_linear(torch.cat([next_context, self.src_mark_embedding(next_src_marks)], dim=-1)), dim=-1)  
                    dst_mark_dist = Categorical(logits=dst_mark_logits)
                    next_dst_marks = dst_mark_dist.sample()  # (batch_size, 1)
                    dst_marks = torch.cat([dst_marks, next_dst_marks], dim=1)
                else:
                    dst_marks = None
            else:
                src_marks = None
                
            # Compute meta, if necessary
            if self.num_meta_classes > 0:
                new_meta = compute_meta(inter_times, self.meta_type)
                new_meta_emb = self.meta_embedding(new_meta)
                metas = torch.cat([metas, new_meta], dim=1)
            else: 
                metas = None
            curr_t = t_start + inter_times.sum(-1).min()
            if curr_t >= t_end:
                time.sleep(5)
                generated = True
            
            batch = Batch(inter_times=inter_times, mask=torch.ones_like(inter_times), src_marks=src_marks, dst_marks = dst_marks, meta=metas)
            features = self.get_features(batch)  # (batch_size, seq_len, num_features)
            context = self.get_context(features, remove_last=False)  # (batch_size, seq_len, context_size)
            next_context = context[:, [-1], :]  # (batch_size, 1, context_size)
            if self.num_meta_classes > 0:
                next_meta_context = torch.cat([next_context, new_meta_emb], dim=-1)

        arrival_times = inter_times.cumsum(-1) + t_start  # (batch_size, seq_len)
        inter_times = diff(arrival_times.clamp(max=t_end), dim=-1)
        mask = (arrival_times <= t_end).float()  # (batch_size, seq_len)
        if self.use_src_marks:
            src_marks = src_marks * mask  # (batch_size, seq_len)
            if self.use_dst_marks:
                dst_marks = dst_marks * mask  # (batch_size, seq_len)
            
        return Batch(inter_times=inter_times, mask=mask, src_marks=src_marks, dst_marks = dst_marks, meta=metas), inter_times.sum(-1).min().item()
    
    def sample1(self, batch_size: int = 1, history: torch.Tensor = None) -> dpp.data.Batch:
        """Generate one event from the model.

        Args:
            context_init: Context vector for the event.
                Used to condition the generator on past events,
                shape (context_size,)

        Returns;
            batch: Batch of sampled sequences. See dpp.data.batch.Batch.
        """
        
        def compute_meta(inter_times, meta_type='basic'):
            if len(inter_times) > 0:
                ts = t_start + inter_times.sum(-1).item()
                # 1848 of hours of sequences were thrown away at the start of the Enron dataset
                date = datetime(1999, 3, 4, 9, 39).astimezone() + pd.to_timedelta(ts + 1848,'h')
                hour = int(date.strftime("%H"))
                weekday = date.weekday()
                week = int(date.strftime("%V"))
                year = int(date.strftime("%Y")) - 1999
                if meta_type == '247':
                    meta = torch.tensor([[7 * weekday + hour]])
                elif meta_type == 'tempo':
                    meta = torch.tensor([[get_tempo_meta(week, year)]])
                elif meta_type == 'basic':    
                    meta = torch.tensor([[get_meta(hour, weekday)]])
            else:
                meta = torch.tensor([[0]])
            return meta

        def get_meta(hour, weekday):
            if weekday in range(7):
                # It's a weekday
                if hour in range(10, 16):
                    # Regular office hours
                    return 0
                elif hour in [7, 8, 9, 16, 17, 18, 19, 20]:
                    # Shoulder
                    return 1
                else:
                    # Non-work hours
                    return 2
            else:
                # It's the weekend
                return 3
    
        def get_tempo_meta(week, year):
            return year * 52 + week - 9
        
        if history is None:
            # Use the default context vector
            context_init = self.context_init
            next_context = context_init[None, None, :].expand(batch_size, 1, -1)
        else:
            # Use the provided history
            batch = Batch(inter_times=inter_times, mask=torch.ones_like(inter_times), src_marks=src_marks, dst_marks = dst_marks, meta=metas)
            features = self.get_features(batch)  # (batch_size, seq_len, num_features)
            context_init = self.get_context(features, remove_last=False)  # (batch_size, seq_len, context_size)
            next_context = context[:, [-1], :]  # (batch_size, 1, context_size)

        inter_times = torch.empty(batch_size, 0)
        if self.use_src_marks:
            src_marks = torch.empty(batch_size, 0, dtype=torch.long)
            if self.use_dst_marks:
                dst_marks = torch.empty(batch_size, 0, dtype=torch.long)
                
        if self.num_meta_classes > 0:
            metas = torch.empty(batch_size, 0, dtype=torch.long)    
            meta_emb = self.meta_embedding(torch.ones(batch_size, 1, dtype=torch.long))
            next_meta_context = torch.cat([next_context, meta_emb], dim=-1)
            
        generated = False
        i = 0
        while not generated and inter_times.sum(-1).min() < t_end:
            inter_time_dist = self.get_inter_time_dist(next_meta_context)
            next_inter_times = inter_time_dist.sample()  # (batch_size, 1)
            inter_times = torch.cat([inter_times, next_inter_times], dim=1)  # (batch_size, seq_len)
            i += 1
            if i%1000 == 0:
                print(i, inter_times.sum(-1).min())
            
            # Generate marks, if necessary
            if self.use_src_marks:
                src_context = self.src_mark_linear1(next_context)
                lin1src_context = F.tanh(src_context)
                lin2src_context = self.src_mark_linear2(lin1src_context)
                src_mark_logits = torch.log_softmax(lin2src_context, dim=-1)
                src_mark_dist = Categorical(logits=src_mark_logits)
                next_src_marks = src_mark_dist.sample()  # (batch_size, 1)
                src_marks = torch.cat([src_marks, next_src_marks], dim=1)
                if self.use_dst_marks:
                    dst_mark_logits = torch.log_softmax(self.dst_mark_linear(torch.cat([next_context, self.src_mark_embedding(next_src_marks)], dim=-1)), dim=-1)  
                    dst_mark_dist = Categorical(logits=dst_mark_logits)
                    next_dst_marks = dst_mark_dist.sample()  # (batch_size, 1)
                    dst_marks = torch.cat([dst_marks, next_dst_marks], dim=1)
                else:
                    dst_marks = None
            else:
                src_marks = None
                
            # Compute meta, if necessary
            if self.num_meta_classes > 0:
                new_meta = compute_meta(inter_times, self.meta_type)
                new_meta_emb = self.meta_embedding(new_meta)
                metas = torch.cat([metas, new_meta], dim=1)
            else: 
                metas = None
            curr_t = t_start + inter_times.sum(-1).min()
            if curr_t >= t_end:
                time.sleep(5)
                generated = True
            
            batch = Batch(inter_times=inter_times, mask=torch.ones_like(inter_times), src_marks=src_marks, dst_marks = dst_marks, meta=metas)
            features = self.get_features(batch)  # (batch_size, seq_len, num_features)
            context = self.get_context(features, remove_last=False)  # (batch_size, seq_len, context_size)
            next_context = context[:, [-1], :]  # (batch_size, 1, context_size)
            if self.num_meta_classes > 0:
                next_meta_context = torch.cat([next_context, new_meta_emb], dim=-1)

        arrival_times = inter_times.cumsum(-1) + t_start  # (batch_size, seq_len)
        inter_times = diff(arrival_times.clamp(max=t_end), dim=-1)
        mask = (arrival_times <= t_end).float()  # (batch_size, seq_len)
        if self.use_src_marks:
            src_marks = src_marks * mask  # (batch_size, seq_len)
            if self.use_dst_marks:
                dst_marks = dst_marks * mask  # (batch_size, seq_len)
            
        return Batch(inter_times=inter_times, mask=mask, src_marks=src_marks, dst_marks = dst_marks, meta=metas), inter_times.sum(-1).min().item()