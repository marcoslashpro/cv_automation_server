from pydantic import BaseModel, Field, EmailStr
from dataclasses import dataclass
import datetime
from uuid import UUID


class UserPublic(BaseModel):
  username: str
  hashed_password: str

class TextFormatterOutput(BaseModel):
  job_title: str = Field(description="The job title presented in the job listing")
  company_name: str = Field(description="The company name presented in the job listing")
  requirements: list[str] = Field(description="The requirements presented in the job listing")
  url: str = Field(description="The original url of the job listing")

class SkillBase(BaseModel):
  skill_name: str
  experience_level: str = Field(
    description=(
      "A simple string describing the experience level of the user with a specific skill."
      " This can be either `Beginner`, `Intermediate`, `Professional`"
    )
  )
  category: str | None = Field(
    default=None,
    description=(
      "A simple description of the skill's category"
      "it can be either `Soft Skill` or `Technical Skill`."
    )
  )

class SkillCreate(SkillBase):
  pass

class SkillModel(SkillBase):
  pass

class SkillPublic(SkillBase):
  id: UUID

class PIIBase(BaseModel):
  email: EmailStr
  first_name: str
  last_name: str
  phone_number: str | None = None
  address: str | None = None
  linkedin_url: str | None = None
  github_url: str | None = None
  personal_website: str | None = None

class PIICreate(PIIBase):
  pass

class PIIModel(PIIBase):
  pass

class PIIPublic(PIIBase):
  id: UUID

class WorkExperienceBase(BaseModel):
  title: str
  company: str
  description: str
  start_date: datetime.datetime
  end_date: datetime.datetime | None = None
  employment_type: str | None = None
  city: str | None = None
  country: str | None = None
  remote: bool | None = None

class WorkExperienceCreate(WorkExperienceBase):
  pass

class WorkExperienceModel(WorkExperienceBase):
  pass

class WorkExperiencePublic(WorkExperienceBase):
  id: UUID

class EducationBase(BaseModel):
  degree: str
  institution: str
  start_date: datetime.datetime = Field(description="Insert the date in ISO format (YYYY-MM)")
  end_date: datetime.datetime = Field(description="Insert the date in ISO format (YYYY-MM)")

class EducationCreate(EducationBase):
  pass

class EducationModel(EducationBase):
  pass

class EducationPublic(EducationBase):
  id: UUID

class CertificationBase(BaseModel):
  name: str
  issuer: str
  issue_date: datetime.datetime | None = None
  expiration_date: datetime.datetime | None = None
  credential_url: str | None = None

class CertificationCreate(CertificationBase):
  pass

class CertificationModel(CertificationBase):
  pass

class CertificationPublic(CertificationBase):
  id: UUID

class ProjectBase(BaseModel):
  title: str
  description: str
  technologies: str | None = None
  link: str | None = None

class ProjectCreate(ProjectBase):
  pass

class ProjectModel(ProjectBase):
  pass

class ProjectPublic(ProjectBase):
  id: UUID

class LanguageBase(BaseModel):
  language: str 
  proficiency_level: str

class LanguageCreate(LanguageBase):
  pass

class LanguageModel(LanguageBase):
  pass

class LanguagePublic(LanguageBase):
  id: UUID

class CVOutput(BaseModel):
  pii: PIIModel | None
  candidate_description: str | None
  experience: list[WorkExperienceModel] | None
  skills: list[SkillModel] | None
  hobbies: list[str] | None
  certifications: list[CertificationModel] | None
  educations: list[EducationModel] | None
  projects: list[ProjectModel] | None
  languages: list[LanguageModel] | None

@dataclass
class EnrichersCrewDeps:
  extractor_output: CVOutput
  job_listing: TextFormatterOutput
  company_report: str
  model_creativity: float
  model_tone: str
  model_instructions: str | None

class Headers(BaseModel):
  full_name: str | None
  email: EmailStr | None
  address: str | None
  date: datetime.datetime | datetime.date = Field(
    exclude=True,
    default_factory=datetime.datetime.today().date
  )

class CVLBody(BaseModel):
  greetings: str
  main_body: str
  sign_off: str = Field(
    default='Sincerely'
  )

class CVLOutput(BaseModel):
  headers: Headers
  body: CVLBody

class EnrichersCrewOutput(BaseModel):
  final_formatted_cv: CVOutput
  final_formatted_cvl: CVLOutput
  company_name: str

