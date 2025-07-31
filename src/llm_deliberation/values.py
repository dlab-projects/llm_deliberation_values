from pydantic import BaseModel, Field


class Values(BaseModel):
    answers: list[str]


def collapse_values(flat_values, messages):
    """
    Reshape a flat list of values into a nested list of lists, matching the shape of messages.

    Args:
        flat_values: List[Any] -- flat output from batch run, length = total messages
        messages: List[List[Any]] -- original nested messages; only the shape is used

    Returns:
        List[List[Any]] -- nested structure matching messages
    """
    output = []
    i = 0
    for msg_list in messages:
        inner = []
        for _ in msg_list:
            if i >= len(flat_values):
                raise ValueError("Not enough flat_values to fill out the messages structure.")
            inner.append(flat_values[i])
            i += 1
        output.append(inner)
    if i != len(flat_values):
        raise ValueError("Some flat_values left unused; messages structure does not match flat_values length.")
    return output


values = [
   "Trust creation and maintenance",
   "Constructive dialogue",
   "Respect and dignity",
   "Professional ethics and integrity",
   "Social etiquette",
   "Religious respect and accommodation",
   "Linguistic respect and inclusivity",
   "Cultural understanding and respect",
   "Cultural heritage and tradition",
   "Financial wellbeing",
   "Sexual freedom and pleasure",
   "Protection of self and others from harm",
   "Environmental consciousness",
   "Authentic expression",
   "Workplace boundaries",
   "Parental care",
   "Consumer and client protection",
   "Child welfare",
   "Animal and pet welfare",
   "Worker welfare and dignity",
   "Workplace etiquette and respect",
   "Economic justice and fairness",
   "Healthcare equity and access",
   "Consent and personal boundaries",
   "Property rights protection",
   "Personal autonomy",
   "Emotional safety and support",
   "Mental health sensitivity and support",
   "Power dynamics values",
   "Privacy and confidentiality",
   "Religious and spiritual authenticity",
   "Emotional intelligence and regulation",
   "Emotional intimacy",
   "Prosocial altruism",
   "Honest communication",
   "Intergenerational respect and relationships",
   "Supportive and caring relationships",
   "Family bonds and cohesion",
   "Conflict resolution and reconciliation",
   "Public good and community engagement",
   "Accessibility",
   "Reciprocal relationship quality",
   "Environmental consciousness",
   "Empathy and understanding",
   "Personal growth",
   "Achievement and recognition",
   "Balance and moderation",
   "Physical health and wellbeing",
   "Personal accountability and responsibility"
]
