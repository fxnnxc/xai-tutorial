import torch 
import torch.nn.functional as F
import time 

def compute_entropy(x):
    h = -1 * (F.softmax(x, dim=-1) * F.log_softmax(x, dim=-1)).sum(dim=-1)
    return h 

class LossManager():    

    def __init__(self):
        self.running_loss = 0 
        self.running_ce_loss = 0 
        self.running_pos_ent_loss = 0
        self.running_neg_ent_loss = 0
        
        self.running_ce_loss_pos = 0 
        self.running_ce_loss_neg = 0
        
    def normal(self, 
                flags, 
                writer,  
                pbar,
                Y_pos,
                Y_hat_pos, 
                Y_hat_neg, 
                energy_pos, 
                energy_neg,
                 **kwrags):
        ce_loss = flags.ce_coeff * torch.nn.CrossEntropyLoss()(Y_hat_pos, Y_pos)  #CE
        loss = ce_loss
        
        flags.count += 1
        writer.add_scalar(f"Loss/CE", ce_loss.item(), flags.count)      
        writer.add_scalar(f"Loss/Loss", loss.item(), flags.count)      
        self.running_loss += loss.item()
        self.running_ce_loss += ce_loss.item()
        
        duration = time.strftime("%H:%M:%S", time.gmtime(time.time()-flags.start_time))         
        pbar.set_description(f"[INFO] : Data:{flags.data} Loss:{flags.loss} |🍀{flags.save_path}|📦#Batch:({flags.count:.2E}) E:({ flags.epoch/flags.epoch_num *100:.2f}%) D:({duration})|"+ 
                            f"Loss :{self.running_loss/flags.count:.3E}| CE :{self.running_ce_loss/flags.count:.3E}|"+  
                            f"|ACC:{flags.current_performance:.3f}")    
        return loss 
    
    
    def default(self,
                flags, 
                writer,  
                pbar,
                Y_pos,
                Y_hat_pos, 
                Y_hat_neg, 
                energy_pos, 
                energy_neg,
                **kwrgs):
        ce_loss = flags.ce_coeff * torch.nn.CrossEntropyLoss()(Y_hat_pos, Y_pos)  #CE
        entropy_pos = compute_entropy(Y_hat_pos).mean()
        entropy_neg = compute_entropy(Y_hat_neg).mean()
        entropy_pos_loss = - flags.pos_ent_coeff * entropy_pos  # Increase the entropy for Positive
        entropy_neg_loss = flags.neg_ent_coeff * entropy_neg    # Decrease the entropy for Negative

        loss = ce_loss + entropy_neg_loss + entropy_pos_loss
        
        flags.count += 1
        writer.add_scalar(f"Loss/CE", ce_loss.item(), flags.count)      
        writer.add_scalar(f"Loss/PosEnt", entropy_pos.item(), flags.count)      
        writer.add_scalar(f"Loss/NegEnt", entropy_neg.item(), flags.count)      
        writer.add_scalar(f"Loss/Loss", loss.item(), flags.count)      
        self.running_loss += loss.item()
        self.running_ce_loss += ce_loss.item()
        self.running_pos_ent_loss+= entropy_pos_loss.item()
        self.running_neg_ent_loss += entropy_neg_loss.item()
        
        duration = time.strftime("%H:%M:%S", time.gmtime(time.time()-flags.start_time))         
        pbar.set_description(f"[INFO] : Data:{flags.data} Loss:{flags.loss} |🍀{flags.save_path}|📦#Batch:({flags.count:.2E}) E:({ flags.epoch/flags.epoch_num *100:.2f}%) D:({duration})|"+ 
                            f"Loss :{self.running_loss/flags.count:.3E}| CE :{self.running_ce_loss/flags.count:.3E}|"+  
                            f"Pos :{self.running_pos_ent_loss/flags.count:.3E}| Neg :{self.running_neg_ent_loss/flags.count:.3E}| ACC:{flags.current_performance:.3f}")    

        return loss 
    
    def energy(self,
                flags, 
                writer,  
                pbar,
                Y_pos,
                Y_hat_pos, 
                Y_hat_neg, 
                energy_pos, 
                energy_neg,
                **kwrgs):
        # No positive entropy penalty + Weighted Negative
        ce_loss = flags.ce_coeff * torch.nn.CrossEntropyLoss()(Y_hat_pos, Y_pos)  #CE
        entropy_neg = compute_entropy(Y_hat_neg)
        entropy_neg_loss = flags.neg_ent_coeff * entropy_neg * 1/(1+(energy_pos - energy_neg).exp()).mean()   # Decrease the entropy for Negative
        loss = ce_loss + entropy_neg_loss 
        
        flags.count += 1
        writer.add_scalar(f"Loss/CE", ce_loss.item(), flags.count)      
        writer.add_scalar(f"Loss/NegEnt", entropy_neg.item(), flags.count)      
        writer.add_scalar(f"Loss/Loss", loss.item(), flags.count)      
        self.running_loss += loss.item()
        self.running_ce_loss += ce_loss.item()
        self.running_neg_ent_loss += entropy_neg_loss.item()
        
        duration = time.strftime("%H:%M:%S", time.gmtime(time.time()-flags.start_time))         
        pbar.set_description(f"[INFO] : Data:{flags.data} Loss:{flags.loss} |🍀{flags.save_path}|📦#Batch:({flags.count:.2E}) E:({ flags.epoch/flags.epoch_num *100:.2f}%) D:({duration})|"+ 
                            f"Loss :{self.running_loss/flags.count:.3E}| CE :{self.running_ce_loss/flags.count:.3E}|"+  
                            f"Pos :{self.running_pos_ent_loss/flags.count:.3E}| Neg :{self.running_neg_ent_loss/flags.count:.3E}| ACC:{flags.current_performance:.3f}")    

        return loss 

    def pseudo_label(self,
                flags, 
                writer,  
                pbar,
                Y_pos,
                Y_hat_pos, 
                Y_hat_neg, 
                energy_pos, 
                energy_neg,
                pseudo_label_alpha, 
                **kwrgs):
        # No positive entropy penalty + Weighted Negative
        ce_loss_pos = flags.ce_coeff * torch.nn.CrossEntropyLoss()(Y_hat_pos, Y_pos)  #CE
        ce_loss_neg = pseudo_label_alpha * flags.ce_coeff * torch.nn.CrossEntropyLoss()(Y_hat_neg, Y_pos)  #CE
        loss = ce_loss_pos + ce_loss_neg
        
        flags.count += 1
        writer.add_scalar(f"Loss/CE_pos", ce_loss_pos.item(), flags.count)      
        writer.add_scalar(f"Loss/CE_neg", ce_loss_neg.item(), flags.count)      
        writer.add_scalar(f"Loss/Loss", loss.item(), flags.count)      
        self.running_loss += loss.item()
        self.running_ce_loss_pos += ce_loss_pos.item()
        self.running_ce_loss_neg += ce_loss_neg.item()
        
        duration = time.strftime("%H:%M:%S", time.gmtime(time.time()-flags.start_time))         
        pbar.set_description(f"[INFO] : Data:{flags.data} Loss:{flags.loss} |🍀{flags.save_path}|📦#Batch:({flags.count:.2E}) E:({ flags.epoch/flags.epoch_num *100:.2f}%) D:({duration})|"+ 
                            f"Loss :{self.running_loss/flags.count:.3E}| CE Pos:{self.running_ce_loss_pos/flags.count:.3E}|"+  
                            f"CE Neg:{self.running_ce_loss_neg/flags.count:.3E}|"+ 
                            f" ACC:{flags.current_performance:.3f}")    

        return loss 

