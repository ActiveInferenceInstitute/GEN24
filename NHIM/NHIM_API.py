"""NHIM — experimental cognitive-mechanisms module.

NHIM is a research sketch of a problem-solving agent: it interprets an
inquiry, gates attention over its features, stores state in a Hopfield memory,
updates beliefs and a bounded distress level, selects a response mode, and
generates a response.

The agent's discrete model — belief update, expected-free-energy policy
selection, and distress dynamics — is specified in
``NHIM/nhim_inquiry_agent.gnn.md`` (Generalized Notation Notation, GNN v1) and
implemented in ``NHIM/nhim_model.py``. This module delegates to that
implementation, so the model has a single home; the specification is validated
against the pinned upstream parser by ``scripts/validate_gnn_spec.py``.
Attention, memory, spaCy interpretation, and the optional GPT-2 question
answering are integration components of the sketch, not part of the model.

Heavy dependencies (spaCy, Transformers, Matplotlib) are imported lazily:
importing this module requires only numpy. The interactive loop is available
through ``python NHIM/NHIM_API.py`` and needs the model cache described in
``docs/nhim.md``.
"""

import logging
import time
import zlib

import numpy as np

import nhim_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Lazily loaded integration components (False = "tried and unavailable").
_nlp = None
_qa_pipeline = None
_gpt2_model = None
_gpt2_tokenizer = None


def get_nlp():
    """Load the spaCy model on first use; return None when unavailable."""
    global _nlp
    if _nlp is None:
        try:
            import spacy

            _nlp = spacy.load("en_core_web_sm")
        except Exception as exc:  # missing package or model download
            logging.warning("spaCy model unavailable (%s); using a token split", exc)
            _nlp = False
    return _nlp or None


def get_qa_pipeline():
    """Load the question-answering pipeline on first use, or return None."""
    global _qa_pipeline
    if _qa_pipeline is None:
        try:
            from transformers import pipeline

            _qa_pipeline = pipeline(
                "question-answering", model="distilbert-base-cased-distilled-squad"
            )
        except Exception as exc:
            logging.warning("QA pipeline unavailable (%s)", exc)
            _qa_pipeline = False
    return _qa_pipeline or None


def get_gpt2():
    """Load GPT-2 and its tokenizer on first use, or return (None, None)."""
    global _gpt2_model, _gpt2_tokenizer
    if _gpt2_model is None:
        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer

            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            _gpt2_model, _gpt2_tokenizer = model, tokenizer
        except Exception as exc:
            logging.warning("GPT-2 unavailable (%s)", exc)
            _gpt2_model, _gpt2_tokenizer = False, False
    return (_gpt2_model or None), (_gpt2_tokenizer or None)


class ResponseGenerator:
    """Generates responses with GPT-2."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt):
        """Generate a response for the prompt."""
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            attention_mask = inputs.ne(self.tokenizer.pad_token_id)
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                max_length=100,
                num_return_sequences=1,
                temperature=0.7,
                top_k=50,
                do_sample=True,
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as exc:
            logging.error("Failed to generate response: %s", exc)
            return "Sorry, I couldn't process that request."


class AttentionalSelection:
    """Bottom-up and top-down attentional gating."""

    @staticmethod
    def bottom_up_attention(sensory_input, feature_weights):
        """Compute the saliency map for the sensory input."""
        try:
            sensory_input = np.atleast_2d(sensory_input)
            feature_weights = np.atleast_2d(feature_weights)
            return np.dot(sensory_input, feature_weights.T)
        except Exception as exc:
            logging.error("Bottom-up attention calculation failed: %s", exc)
            return None

    @staticmethod
    def top_down_attention(saliency_map, goal_probability):
        """Modulate the saliency map by the goal probability."""
        try:
            return saliency_map * goal_probability
        except Exception as exc:
            logging.error("Top-down attention modulation failed: %s", exc)
            return None


class HopfieldNetwork:
    """Hopfield network used as an associative memory."""

    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        """Store the given patterns (Hebbian outer-product rule)."""
        try:
            patterns = [np.asarray(p, dtype=float).ravel() for p in patterns]
            for pattern in patterns:
                self.weights += np.outer(pattern, pattern)
            self.weights /= len(patterns)
            np.fill_diagonal(self.weights, 0)  # no self-connections
        except Exception as exc:
            logging.error("Training Hopfield network failed: %s", exc)

    def retrieve(self, pattern, steps=10):
        """Retrieve the stored pattern closest to the cue."""
        try:
            pattern = np.asarray(pattern, dtype=float).ravel()
            state = np.sign(pattern)
            state[state == 0] = 1
            for _ in range(steps):
                state = np.sign(self.weights @ state)
                state[state == 0] = 1
            return state
        except Exception as exc:
            logging.error("Memory retrieval failed: %s", exc)
            return None


class MemorySystem:
    """Short-term store backed by a Hopfield network."""

    def __init__(self, capacity, hopfield_size):
        self.capacity = capacity
        self.memory_storage = []
        self.hopfield_network = HopfieldNetwork(size=hopfield_size)

    def memory_encoding(self, sensory_information):
        """Encode sensory information into memory using Hebbian learning."""
        try:
            encoded_memory = self._apply_hebbian_learning_rule(
                np.asarray(sensory_information, dtype=float).ravel()
            )
            self.memory_storage.append(encoded_memory)
            if len(self.memory_storage) > self.capacity:
                self.memory_storage.pop(0)
            self.hopfield_network.train(self.memory_storage)
        except Exception as exc:
            logging.error("Memory encoding failed: %s", exc)

    def memory_retrieval(self, retrieval_cues):
        """Retrieve a memory from the Hopfield network using the given cues."""
        try:
            cues = np.asarray(retrieval_cues, dtype=float).ravel()
            if len(cues) != self.hopfield_network.size:
                cues = np.resize(cues, self.hopfield_network.size)
            return self.hopfield_network.retrieve(cues)
        except Exception as exc:
            logging.error("Memory retrieval failed: %s", exc)
            return None

    @staticmethod
    def _apply_hebbian_learning_rule(data, other_data=None):
        """Apply the Hebbian learning rule to the input data."""
        eta = 0.1  # learning rate
        if other_data is None:
            other_data = data
        return eta * data * other_data


class ProblemSolvingIntelligence:
    """The NHIM inquiry agent.

    Beliefs, policy selection, and distress come from the GNN-specified model
    in ``nhim_model``; interpretation, attention, memory, and response
    generation are the sketch's integration components.
    """

    def __init__(self, response_generator=None):
        start_time = time.time()
        self.memory_system = MemorySystem(
            capacity=100, hopfield_size=nhim_model.NUM_STATES
        )
        self.response_generator = response_generator
        self.beliefs = nhim_model.initial_beliefs()
        self.distress = 0.0
        self.last_action = 0
        self.step_index = 0
        self.context_history = []
        logging.info("Initialization time: %s seconds", time.time() - start_time)

    def interpret_inquiry(self, inquiry):
        """Return ``(entities, doc)``; spaCy entities when the model is loaded."""
        nlp = get_nlp()
        if nlp is None:
            return [(token, "TOKEN") for token in inquiry.split()], None
        doc = nlp(inquiry)
        return [(ent.text, ent.label_) for ent in doc.ents], doc

    @staticmethod
    def observe(inquiry, entities):
        """Map an inquiry to a sense-data category (deterministic).

        Uses ``zlib.crc32`` rather than ``hash()`` so the observation is stable
        across processes. The category is a placeholder observation model for
        the sketch; the generative model's likelihood ``A`` is what relates
        contexts to sense data.
        """
        key = "|".join(text for text, _ in entities) or inquiry
        return int(zlib.crc32(key.encode("utf-8")) % nhim_model.NUM_OBSERVATIONS)

    def process_inquiry(self, inquiry):
        """Process one inquiry; return ``(response, beliefs, distress)``."""
        try:
            entities, doc = self.interpret_inquiry(inquiry)
            logging.debug("Entities: %s", entities)

            # Attention gating over the inquiry's numeric features.
            numeric_features = np.array(
                [float(zlib.crc32(text.encode("utf-8")) % 1000) for text, _ in entities]
                or [0.0]
            ).reshape(-1, 1)
            feature_weights = np.ones(numeric_features.shape[1])
            saliency_map = AttentionalSelection.bottom_up_attention(
                numeric_features, feature_weights
            )
            goal_probability = np.ones(len(saliency_map))
            modulated_saliency = AttentionalSelection.top_down_attention(
                saliency_map, goal_probability
            )
            logging.debug("Modulated saliency: %s", modulated_saliency)

            # The GNN-specified model: perception, policy, distress.
            observation = self.observe(inquiry, entities)
            self.beliefs = nhim_model.update_beliefs(
                self.beliefs, observation, self.last_action
            )
            self.distress = nhim_model.update_distress(
                self.distress, self.beliefs, self.step_index
            )
            policy = nhim_model.policy_posterior(self.beliefs)
            action = nhim_model.select_action(self.beliefs)
            self.last_action = action

            self.memory_system.memory_encoding(self.beliefs)
            response = self.generate_response(action, self.beliefs, inquiry, doc)
            self.context_history.append(inquiry)
            self.step_index += 1

            logging.debug(
                "Beliefs: %s, distress: %.4f, policy: %s, action: %s",
                self.beliefs,
                self.distress,
                policy,
                nhim_model.ACTION_LABELS[action],
            )
            return response, self.beliefs, self.distress
        except Exception as exc:
            logging.error("Failed to process inquiry: %s", exc)
            return "Error processing inquiry", self.beliefs, self.distress

    @staticmethod
    def is_question(doc):
        """Return True when the spaCy document's root is a question."""
        if doc is None:
            return False
        for token in doc:
            if token.dep_ == "ROOT" and token.tag_ in ("WP", "WRB", "VBZ", "VBP"):
                return True
        return False

    def answer_question(self, inquiry):
        """Answer a question from recent context with the QA pipeline."""
        qa_pipeline = get_qa_pipeline()
        if qa_pipeline is None:
            return None
        context = " ".join(self.context_history[-5:]) or "General information"
        result = qa_pipeline(question=inquiry, context=context)
        return f"Answer: {result['answer']}"

    def generate_response(self, action, beliefs, inquiry="", doc=None):
        """Generate a response for the selected response mode."""
        if self.is_question(doc):
            answer = self.answer_question(inquiry)
            if answer:
                return answer
        generator = self.response_generator
        if generator is None:
            model, tokenizer = get_gpt2()
            if model is not None:
                generator = self.response_generator = ResponseGenerator(model, tokenizer)
        if generator is not None:
            return generator.generate(inquiry)
        return f"Action: {nhim_model.ACTION_LABELS[action]}, Beliefs: {beliefs}"


def validate_model(psi, inquiries):
    """Process a list of inquiries and plot beliefs, distress, and timing."""
    results = []
    times = []
    belief_history = []
    distress_history = []

    for inquiry in inquiries:
        start_time = time.time()
        initial_beliefs = psi.beliefs.copy()
        initial_distress = psi.distress
        response, updated_beliefs, updated_distress = psi.process_inquiry(inquiry)
        elapsed = time.time() - start_time

        times.append(elapsed)
        belief_history.append(np.asarray(updated_beliefs, dtype=float))
        distress_history.append(float(updated_distress))
        results.append(
            {
                "inquiry": inquiry,
                "initial_beliefs": initial_beliefs,
                "final_beliefs": updated_beliefs,
                "initial_distress": initial_distress,
                "final_distress": updated_distress,
                "response": response,
                "elapsed_seconds": elapsed,
            }
        )

    try:
        import matplotlib.pyplot as plt

        belief_history = np.array(belief_history)
        plt.figure(figsize=(12, 8))
        for i in range(belief_history.shape[1]):
            plt.plot(
                range(len(belief_history)),
                belief_history[:, i],
                marker='o',
                label=f"{nhim_model.STATE_LABELS[i]}",
                linewidth=2,
            )
        plt.xlabel('Inquiry Index')
        plt.ylabel('Beliefs')
        plt.title('Beliefs Over Time')
        plt.legend(title='Inquiry context')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.plot(range(len(times)), times, marker='o', label='Processing time')
        plt.plot(range(len(distress_history)), distress_history, marker='x', label='Distress')
        plt.xlabel('Inquiry Index')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as exc:
        logging.warning("Plotting unavailable: %s", exc)

    return results


def handle_user_queries():
    """Run the interactive inquiry loop."""
    psi = ProblemSolvingIntelligence()
    while True:
        user_query = input("Enter your query (or 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        response, updated_beliefs, updated_distress = psi.process_inquiry(user_query)
        print(f"Response: {response}")
        print(f"Updated Beliefs: {updated_beliefs}")
        print(f"Updated Distress: {updated_distress}")


if __name__ == "__main__":
    handle_user_queries()
