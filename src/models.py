from pydantic import BaseModel, Field


class Crew(BaseModel):
    name: str
    size: int = Field(default=0, init=False)
    members: list[str] = Field(default_factory=list)
    adults: list[str]

    def __init__(self, name: str, adults: list[str]):
        super().__init__(name=name, adults=adults, members=adults)
        self.recompute_size()

    def add_member(self, member: str):
        self.members.append(member)
        self.recompute_size()

    def recompute_size(self):
        self.size = len(self.members)


class Center(BaseModel):
    name: str
    crews: list[Crew]
    crew_count: int = Field(default=0, init=False)

    def __init__(self, name: str, crews: list[Crew]):
        super().__init__(name=name, crews=crews)
        self.crew_count = len(crews)

    def add_crew(self, crew: Crew):
        self.crews.append(crew)
        self.crew_count += 1

    def remove_crew(self, crew: Crew):
        self.crews.remove(crew)
        self.crew_count -= 1


class Person(BaseModel):
    name: str
    center: str | None = None
    crew: str | None = None


class Youth(Person):
    year: str
    gender: str
    history: str
    parent_name: str | None = None
    siblings: list[str]
    first_choice: str | None = None
    second_choice: str | None = None
    third_choice: str | None = None
    past_leaders: list[str]
    role: str = 'Youth'  # Can be "Youth" or "Young Adult"

    def __init__(
        self,
        name: str,
        year: str,
        gender: str,
        history: str,
        role: str = 'Youth',
        siblings: str | None = None,
        parent_name: str | None = None,
        first_choice: str | None = None,
        second_choice: str | None = None,
        third_choice: str | None = None,
        past_leaders: list[str] = [],
    ):
        siblings_list: list[str] = []
        if siblings is not None:
            siblings_list = siblings.split('|')

        super().__init__(
            name=name,
            year=year,
            gender=gender,
            history=history,
            role=role,
            siblings=siblings_list,
            parent_name=parent_name,
            first_choice=first_choice,
            second_choice=second_choice,
            third_choice=third_choice,
            past_leaders=past_leaders,
        )


class Adult(Person):
    children: list[Youth]
