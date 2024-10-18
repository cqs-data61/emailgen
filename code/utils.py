import argparse
import numpy as np
import pandas as pd
import pickle
import random
import re
import torch
import spacy
from datetime import datetime

nlp = spacy.load('en_core_web_sm')


def generate_email_body(sender_name, recipient_name, email_thread, model, args, tokenizer):    
    
    # Generate email
    greetings = [f'Hi {recipient_name},\n', f'Hello {recipient_name},\n', f'{recipient_name}, \n']
    chosen_greeting = greetings[np.random.randint(len(greetings))] 

    if len(email_thread) == 0: 
        # Fallback in case, but shouldn't happen
        print("prompt is empty")
        thread_starter = get_email_thread_topic()
        chosen_greeting = chosen_greeting + " " + thread_starter

    prompt_text = email_thread + chosen_greeting

    prefix = args.prefix if args.prefix else args.padding_text
    encoded_prompt = tokenizer.encode(prefix + prompt_text, add_special_tokens=False, return_tensors="pt")
    encoded_prompt = encoded_prompt.to(args.device)

    if encoded_prompt.size()[-1] == 0:
        input_ids = None
    else:
        input_ids = encoded_prompt

    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=args.length + len(encoded_prompt[0]),
        temperature=args.temperature,
        top_k=args.k,
        top_p=args.p,
        repetition_penalty=args.repetition_penalty,
        do_sample=True,
        num_return_sequences=args.num_return_sequences,
    )

    # Remove the batch dimension when returning multiple sequences
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()

    generated_sequences = []

    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

        # Remove all text after the stop token
        text = text[: text.find(args.stop_token) if args.stop_token else None]

        cleaned_text = text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
        sentences = [x for x in nlp(cleaned_text).sents]
        sentences_cleaned = []
        for i in range(len(sentences)):
            if check_sentence(str(sentences[i]), sender_name, recipient_name, i):
                sentences_cleaned.append(sentences[i])
        farewell = ['\n' + f'Regards, {sender_name}.\n ', '\n' + f'Cheers, {sender_name}.\n', 
                 '\n' + f'Best, {sender_name}. \n', '\n' + f'{sender_name}.\n']
        sign_off = farewell[np.random.randint(len(farewell))] 

        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        num_sentences = min(2, len(sentences))
        total_sequence = (
            prompt_text[-len(chosen_greeting): ] + \
                " ".join(str(x).replace("Enron", "Wilson").replace('<eos>','').replace('<epom>','').replace('eos', '').replace('<', '').replace('>', '').strip() for x in sentences[:num_sentences]) + " " + sign_off
        )

        generated_sequences.append(total_sequence)

    # Simple heuristic - just choose the longest... 
    # ToDo: select based on which response gives highest coherence/cohesion.
    generated_email = max(generated_sequences, key=len)

    return generated_email


def run_generation(run_time, model, pp_model, args, tokenizer, employee_name_map, employee_email_map, message_type_props, \
                   subjects="data/data_for_simulation/subjects_by_ID.csv"):
    # Storage structs
    generated_emails = []
    thread_participants_dic = {}
    recip_set_dict = get_recips_dict()
    generated_emails_df = pd.DataFrame(columns=['eid', 'date-time', 'from', 'to', 'email', 'email_thread', \
                                                'thread_length', 'email_type', 'thread_id', 'full_email_thread', 'subject'])
    recipients_df = pd.DataFrame(columns=['eid', 'from', 'to', 'recip_set', 'email_type', 'recipient_type', 'thread_id'])
    
    # Load generated subjects
    generated_subjects_df = pd.read_csv(subjects)
    generated_subjects_dict = generated_subjects_df.groupby('ID')['Subject'].apply(list).to_dict()
#     subject_counter = 0
#     max_subject_count = len(generated_subjects_list) - 1 
    
    # Load counts of emails sent from raw_data
    new_thread_count = 0
    cum_time = 0 # We now need to track this.
    
    # Generate the TPP events
    sampled_batch, t_end = pp_model.sample(0, t_end=run_time, batch_size=1)
    generated = pd.DataFrame(list(zip(sampled_batch['inter_times'][0].detach().cpu().numpy(), 
                                  sampled_batch['src_marks'][0].detach().cpu().numpy(),
                                  sampled_batch['dst_marks'][0].detach().cpu().numpy(),
                                  sampled_batch['meta'][0].detach().cpu().numpy())), 
               columns =['deltas', 'from', 'to', 'meta']) 
    generated['ts'] = generated['deltas'].cumsum() 
    num_emails = len(generated)
    cum_time = 0
    for email_idx in range(num_emails):  
        raw_delta = generated['deltas'].iloc[email_idx]
        sender_id = generated['from'].iloc[email_idx]
        cum_time += raw_delta*60*60
        sender_name = employee_name_map[sender_id]
        
        recip_set = generated['to'].iloc[email_idx]
        if recip_set < 54:
            recipients_list = [recip_set]
        else:    
            recipients_list = list(recip_set_dict[recip_set])
        if len(recipients_list) <= 2:
            recipient_name = ' and '.join(employee_name_map[x] for x in recipients_list)
        else:
            recipient_name = 'all'
        # Choose whether to start new thread or append to existing
        new_thread_prop = message_type_props[message_type_props['from'] == sender_id].iloc[0]['new_thread_prop']
        if random.uniform(0, 1) < new_thread_prop:
            # start new thread
            email_type = 'new_thread'
            subject = random.choice(generated_subjects_dict[sender_id])
            email_thread_string = subject + '\n'
            full_email_thread = ''
            new_thread_count += 1
            thread_id = new_thread_count            
            thread_participants_dic[thread_id] =  [int(sender_id)] + list(recipients_list)

            # Generate email
            try:
                email = generate_email_body(sender_name, recipient_name, email_thread_string, model, args, tokenizer)
            except:
                print(f"problem with email id: {email_idx}. Skipping")
                print(f"Sender: {sender_name}. Recipient: {recipient_name}. Email_thread: {email_thread_string}.")
                email = ''
            email_thread_length = 1
        else:
            # decide if 'reply' or 'fwd' email type
            fwd_prop = message_type_props[message_type_props['from'] == sender_id].iloc[0]['fwd_prop']
            if random.uniform(0, 1) < fwd_prop:
                # 'fwd' email type.
                # Find most recent thread that can be forwarded
                email_type = 'fwd'
                thread_found = False
                for thread_id, v in reversed(list(thread_participants_dic.items())):
                    if (sender_id in v) and not all(elem in v for elem in recipients_list):
                        thread_found = True
                        recipient_ids = [x for x in recipients_list if x not in v]
                        recipient_name = ', '.join(employee_name_map[x] for x in recipient_ids)
                        break
                if thread_found:
                    # Look up recipients_df to find the last email in thread that sender_id received
                    print('thread found')
                    thread = recipients_df[((recipients_df['thread_id'] == thread_id) & (recipients_df['to'] == \
                                                                                         sender_id))].tail(1)
                else:
                    # Fall back to a reply email when an appropriate thread can't be found to fwd
                    # print("Couldn't find an appropriate thread to forward, so will try to reply instead.")
                    email_type = 'reply'
                    thread_found = False
                    for thread_id, v in reversed(list(thread_participants_dic.items())):
                        if (sender_id in v) and all(elem in v for elem in recipients_list):
                            thread_found = True
                            thread = recipients_df[((recipients_df['thread_id'] == thread_id) & (recipients_df['to'] == \
                                                                                         sender_id))].tail(1)
                            break
                    if not thread_found:
                        recipient_id = recipients_list[0]
                        thread = recipients_df[((recipients_df['from'] == recipient_id) & (recipients_df['to'] == sender_id)) | \
                                               ((recipients_df['to'] == recipient_id) & (recipients_df['from'] == \
                                                                                         sender_id))].tail(1)
            else:
                # 'reply' email type
                # check if an appropriate thread exists to add to
                email_type = 'reply'
                thread_found = False
                for thread_id, v in reversed(list(thread_participants_dic.items())):
                    if (sender_id in v) and all(elem in v for elem in recipients_list):
                        thread_found = True
                        thread = recipients_df[((recipients_df['thread_id'] == thread_id) & (recipients_df['to'] == \
                                                                                     sender_id))].tail(1)
                        break
                if not thread_found:
                    recipient_id = recipients_list[0]
                    thread = recipients_df[((recipients_df['from'] == recipient_id) & (recipients_df['to'] == sender_id)) | \
                                           ((recipients_df['to'] == recipient_id) & (recipients_df['from'] == \
                                                                                     sender_id))].tail(1)

            # Generate email
            if (thread.empty): 
                # Fall back to new thread 
                email_type = 'new_thread'
                subject = random.choice(generated_subjects_dict[sender_id])
                email_thread_string = subject + '\n'
                full_email_thread = ''
                new_thread_count += 1
                thread_id = new_thread_count
                thread_participants_dic[thread_id] = [int(sender_id)] + recipients_list
                try:
                    email = generate_email_body(sender_name, recipient_name, email_thread_string, model, args, tokenizer)
                except:
                    print(f"Problem with row: {index}. Skipping.")
                    email = ''
                email_thread_length = 1
            else:
                # reply to existing thread
                # TODO: allow for multiple threads to be considered, not just the most recent
                # Find id of email to fwd/reply to
                email_id = thread.iloc[0]['eid']

                # Retrieved email thread
                email_thread = generated_emails_df[generated_emails_df.eid == email_id]
                email_thread_string = email_thread.iloc[0]['email_thread']
                full_email_thread = email_thread.iloc[0]['full_email_thread']
                thread_id = email_thread.iloc[0]['thread_id']
                subject = email_thread.iloc[0]['subject']
                if email_type == 'fwd':
                    thread_participants_dic[thread_id] = list(set(list(thread_participants_dic[thread_id]) + list(recipients_list)))
                    email = "Hi " + recipient_name + ", " + get_fwd_email() + '\n'
                elif email_type == 'reply': 
                    email_thread_string = subject + '\n' + email_thread_string
                    try:
                        email = generate_email_body(sender_name, recipient_name, email_thread_string, model, args, tokenizer)
                    except:
                        print(f"Problem with email id: {email_idx}. Skipping.")
                        email = ''
                email_thread_length = email_thread.iloc[0]['thread_length'] + 1
        email_thread_string += email + '\n '   
        generated_emails.append([email_idx, email])


        # Write recipients to dataframe
        CC_list = []
        for recipient_id in thread_participants_dic[thread_id]:
            if recipient_id == sender_id:
                continue
            elif recipient_id in recipients_list:
                # In general not ideal to grow a dataframe, however we use it in between each data accumulation step
                if pd.isnull(recipients_df.index.max()):
                    recipients_df.loc[0] = [email_idx, sender_id, recipient_id, recip_set, email_type, 'To', thread_id]
                else: 
                    recipients_df.loc[recipients_df.index.max() + 1] = \
                        [email_idx, sender_id, recipient_id, recip_set, email_type, 'To', thread_id]
            else:
                # TODO: this makes all replies/fwd a "reply-all". Often don't do that for 'fwd'.
                CC_list.append(recipient_id)
                recipients_df.loc[recipients_df.index.max() + 1] = \
                        [email_idx, sender_id, recipient_id, recip_set, email_type, 'CC', thread_id]

        # Write email content to dataframe
        email_with_recipients = '------------------------------------------------\n' + \
        f'Email #: {email_thread_length}.\n' + 'From: ' + employee_email_map[sender_id] + '\n To: '+ \
                                    '; '.join(employee_email_map[x] for x in recipients_list) + '\n' 
        if len(CC_list) > 0:
            email_with_recipients += 'CC: ' + '; '.join(employee_email_map[x] for x in CC_list) + '\n' 
        email_with_recipients += '\n' + email + '\n '
        full_email_thread += email_with_recipients
        ## Datetime processing
        scaled_time = cum_time
        email_dt = datetime(2021, 3, 4, 0, 59).astimezone() + pd.to_timedelta(scaled_time, 's')
        email_dt_str = email_dt.strftime('%a') + "   " + email_dt.strftime('%d/%m/%y %I:%M %p')
        generated_emails_df.loc[email_idx] = \
            [email_idx, email_dt_str, sender_id, recip_set, email, email_thread_string, email_thread_length, email_type, \
             thread_id, full_email_thread, subject] 
        if email_idx % 5000 == 0:
            generated_emails_df.to_csv(str(email_idx)+'emails.csv')
            recipients_df.to_csv(str(email_idx)+'recipients.csv')
    return generated_emails_df, generated_emails, recipients_df



def get_parser(model_type='gpt2', model_name_or_path='data/models/gpt2/mbox_test_results_slim/', 
               prompt="<|startoftopic|> FYI. The report posted on the website ", length='150', k='50',
               num_return_sequences='10'):
    stop_token = '<|endoftext|>'
    temperature = '1.0'
    p = '0.9'
    repetition_penalty = '1.0'
    seed = '42'
    prefix = """Ideally, we needed to receive the Team Selection form
we sent you.  The information needed is then easily transferred into the
database directly from that Excel spreadsheet.  If you do not have the
ability to complete that form, inserting what you listed below, we still
require additional information.

We need each person's email address.  Without the email address, we cannot
email them their internet link and ID to provide feedback for you, nor can
we send them an automatic reminder via email.  It would also be good to have
each person's phone number, in the event we need to reach them."""
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default='gpt2',
        type=str,
    )
    parser.add_argument(
        "--model_name_or_path",
        default='data/models/gpt2/mbox_test_results_slim/',
        type=str,
        help="Path to pre-trained model or shortcut name",
    )

    parser.add_argument("--prompt", type=str, default="<|startoftopic|> FYI. The report posted on the website ")
    parser.add_argument("--length", type=int, default=150)
    parser.add_argument("--stop_token", type=str, default='<|endoftext|>', help="Token at which text generation is stopped")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--p", type=float, default=0.9)

    parser.add_argument("--prefix", type=str, default="Enron.", help="Text added prior to input.")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument(
       "--fp16",
       action="store_true",
       help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    args = parser.parse_args(args=['--model_type', model_type, '--model_name_or_path', model_name_or_path, 
                                   '--prompt', prompt, '--length', length, '--stop_token', stop_token, '--k', k,
                                   '--seed', seed, '--prefix', prefix, '--temperature', temperature,
                                   '--p', p, '--num_return_sequences', num_return_sequences, 
                                   '--repetition_penalty', repetition_penalty])
    return args

def get_recips_dict():
    with open('../data/data_for_simulation/enron_recip_set_dict.pkl', 'rb') as f:
        recip_dict = pickle.load(f)
    return recip_dict

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

        
def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def check_sentence(sentence, sender_name, receiver_name, sentence_number=0):
    # Just a hack  because I couldn't get Spacy's textpipe.doc working, and haven't cleaned training data properly yet
    # Heuristic: throw away sentences with html/emails or other bad artefacts. 
    junk = re.findall(r'<.*?>', sentence)
    junk += re.findall('@enron.com', sentence)
    junk += re.findall("greetings", sentence.lower())
    junk += re.findall("-----Original", sentence)
    junk += re.findall("From:", sentence)
    junk += re.findall("http://", sentence)
    junk += re.findall('Subject', sentence)
    junk += re.findall('Message---', sentence)
    junk += re.findall('style=', sentence)
    junk += re.findall('align=', sentence)
    junk += re.findall("-------------", sentence)
    junk += re.findall("Forwarded by", sentence)
    junk += re.findall('href=', sentence)
    junk += re.findall(sender_name.lower(), sentence.lower())
    if sentence_number > 0:    
        junk += re.findall(receiver_name.lower(), sentence.lower())
    if len(junk) > 0:
        return False
    elif sentence_number == 0 and sentence.lower().count(receiver_name.lower()) > 1:
        return False
    return True


def get_email_thread_topic():
    # Randomly select from a number of different thread starters.
    # TODO: Choose thread topic based on Topic Model topics of the sender.

    topics = [
        "Attached is the letter that we sent to Lynch explaining the info ",
        "Here is a spreadsheet detailing our plans for ",
        "I will be out of the office on Thursday. ",
        "Can you give me more details on ", 
        "How is progress going on creating the spreadsheets? You will probably ", 
        "Please plan to attend a meeting on Friday 3pm to discuss the selection of ",
        "I am sending off a follow-up to a bid I submitted to ",
        "Attached are initial comments of ",
        "The target date by which to update these ",
        "I would like you to direct Wade on ",
        "I think the presentation looks good.  It would be useful to ",
        "Please review the attached draft Enron comments in response to ",
        "Did you get set up on the account?  Try and "
    ]
    
    topic_id = np.random.randint(0, len(topics))
    return topics[topic_id]


def get_fwd_email():
    # Randomly select from a number of different email starters.

    topics = [
        "Please get back to me with your thoughts by end of day.",
        "What do you think?",
        "Have you seen this?",
        "Can you help out here?",
        "FYI."
    ]
    
    topic_id = np.random.randint(0, len(topics))
    return topics[topic_id]