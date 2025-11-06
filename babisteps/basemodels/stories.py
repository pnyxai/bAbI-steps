from pydantic import BaseModel

from babisteps.basemodels.FOL import FOL


class Story(BaseModel):
    world_enumerate: list[FOL]
    describe_len: int
    story: list[FOL]
    question: str
    # Union can be a string or a list of strings
    answer: str | list[str]
    response_templates: dict

    def create_json(self):
        dict_json = {}
        dict_json["idx"] = 0
        dict_json["question"] = self.question
        dict_json["answer"] = self.answer
        wd = ""
        # Each FOL in world_enumerate and story to nl
        for wd_i in self.world_enumerate:
            wd = wd + wd_i.to_nl() + " "
        s = ""
        for s_i in self.story:
            s = s + s_i.to_nl() + "\n"

        dict_json["world_enumerate"] = wd
        dict_json["story"] = s
        return dict_json

    def create_txt(self):
        txt = ""
        for wd_i in self.world_enumerate:
            txt += wd_i.to_nl() + "\n"

        txt += "\n"
        for idx, s_i in enumerate(self.story):
            if idx == self.describe_len:
                txt += "\n"
            txt += s_i.to_nl() + "\n"

        txt += f"\n{self.question}\n"
        if isinstance(self.answer, str):
            txt += f"{self.answer}\n"
        else:
            txt += ", ".join(self.answer) + "\n"
        txt += "-" * 40
        txt += "\n\n"

        return txt
