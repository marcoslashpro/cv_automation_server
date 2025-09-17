from __future__ import annotations

import io
import os
import json
import logging
import tempfile
import requests
import datetime
import subprocess
import streamlit as st
from pathlib import Path
from uuid import uuid4, UUID
from dotenv import load_dotenv
from jinja2 import TemplateError
from docxtpl import DocxTemplate
from abc import ABC, abstractmethod
from typing import Any, Literal, TypeVar, Type
from streamlit.runtime.uploaded_file_manager import UploadedFile
from pydantic import BaseModel, Field, ValidationError, computed_field

from cv_automation_models import (
  PIICreate, PIIPublic,
  SkillCreate, SkillPublic,
  ProjectCreate, ProjectPublic,
  LanguageCreate, LanguagePublic,
  EducationCreate, EducationPublic,
  CertificationCreate, CertificationPublic,
  WorkExperienceCreate, WorkExperiencePublic,
  UserPublic, CVOutput, EnrichersCrewOutput, CVLOutput
)

load_dotenv()

# --- SETUP --- #
st.set_page_config(layout = 'wide', page_title="Cv Customizer")

logger = logging.getLogger(__name__)

URL = 'http://107.21.44.255/cv-automation/cv'
USERNAME = os.environ.get("USERNAME", 'halo_user')
INPUT_CV_PATH = Path(__file__).parent / "public/cv_template.docx"
INPUT_CVL_PATH = Path(__file__).parent / "public/_cvl_template.docx"
if (
  not os.path.exists(INPUT_CV_PATH) or not os.path.exists(INPUT_CVL_PATH)
):
  raise RuntimeError(
    "Invalid Paths to the CV and/or to the CVL templates"
  )

user = UserPublic(username=USERNAME, hashed_password='Not Needed')

# --- GENERICS --- #
TCreate = TypeVar("TCreate", bound=BaseModel)
TPublic = TypeVar("TPublic", bound=BaseModel)

# --- HELPERS --- #
def save_to_db(
    endpoint: str,
    user: UserPublic,
    models_create: list[BaseModel],
    model_public: Type[TPublic],
    url: str = URL
  ) -> list[TPublic] | None:
  """
  Send a POST request to a specific endpoint in order to save some personal information.

  Args:
    endpoint: The endpoint where to send the POST request.
    user: A valid UserPublic with username and hashed password.
    models_create: A list of models to insert into the db.
    model_public: The Type of model that we expect back from the server, this is used for validation.
    url: The base url of the server, defaults to the URL constant.

  Returns:
    A list of validated models from the db or nothing if an error occured.
  """
  final_url = f"{url}/{endpoint}"
  payload = {
    "user": user.model_dump(),
    endpoint: [model.model_dump(mode='json') for model in models_create]
  }

  try:
    res = requests.post(final_url, json=payload)
  except requests.RequestException as e:
    st.error(f"Unable to send request to url: {final_url}, error: {str(e)}")
    return

  if res.status_code == 200:
    try:
      if isinstance(res.json(), list):
        return [model_public.model_validate(model) for model in res.json()]
      else:
        return [model_public.model_validate(res.json())]
    except ValidationError as e:
      logger.error(f"Invalid response body from {final_url}: {e.errors(include_url=False)}")
  elif res.status_code == 404:
    st.error(f"{res.status_code} from the server, please check that {final_url} is a valid url.")
  elif res.status_code == 500:
    st.error("Something went wrong on the server side, we apologize for the inconvenience.")
  else:
    st.error(f"Something went wrong while saving information to the database: {res.status_code}")

def get_from_db(
    endpoint: str,
    model: Type[TPublic],
    username: str = USERNAME,
    url: str = URL
  ) -> list[TPublic] | None:
  """
  Sends a GET request to the specified endpoint and returns the response.

  Args:
    endpoint: The specific endpoint to add to a pre-defined url
    model: The type of the model that we expect from the response. This will be used
    for data validation.
    username: The username used to access info in the db, defaults the USERNAME constant.
    url: The base url of the server, defaults to the URL constant.

  Returns:
    A list of validated models from the db or nothing if an error occured.
  """
  final_url = f"{url}/{endpoint}/{username}"
  try:
    res = requests.get(final_url)
  except requests.RequestException as e:
    st.error(f"Unable to send request to url: {final_url}, error: {str(e)}")
    return

  if res.status_code == 200:
    body = res.json()
    try:
      if not body:
        return
      if isinstance(body, list):
        models = [model.model_validate(m) for m in body]
      else:
        models = [model.model_validate(body)]

      return models
    except ValidationError as e:
      logger.error(f"Error while loading {model} from the db: {e.errors(include_url=False)}")
  elif res.status_code == 500:
    st.error("Something went wrong on the server side, we apologize for the inconvenience.")
  else:
    st.error(f"Something went wrong while getting information from the database: {res.status_code}")

def update_in_db(
    endpoint: str,
    models: list[TCreate],
    ids: list[str | UUID],
    public_model: Type[TPublic],
    username: str = USERNAME,
    url: str = URL
  ) -> list[TPublic] | None:
  """
  Send a PATCH request to the given endpoint in order to modify existing ententies in the db.

  Args:
    endpoint: The endpoint where to send the PATCH request.
    models: The models that we want to update in the db.
    ids: The ids of the models. The length of the ids and models must be the same or a IndexError is raised.
    public_model: The Type of the models that we expect back from the db, this is used for validation.
    username: The username used in order to authenticate on the server, defaults to the USERNAME constant.
    url: The base url where to send the request, default to the URL constant.

  Returns:
    A list of validated models from the db or nothing if an error occured.
  """
  if len(ids) != len(models):
    raise IndexError(
      f"The numer of ids and models provided must be the same, instead got {len(ids)} and {len(models)}"
    )

  to_add = [public_model(id=id.hex if isinstance(id, UUID) else id, **m.model_dump()) for id, m in zip(ids, models)]

  final_url = f"{url}/{endpoint}/{username}"
  payload = [model.model_dump(mode='json') for model in to_add]

  try:
    res = requests.patch(final_url, json=payload)
  except requests.RequestException as e:
    st.error(f"Unable to send request to url: {final_url}, error: {str(e)}")
    return

  if res.status_code == 200:
    body = res.json()
    try:
      if isinstance(body, list):
        updated_models = [public_model.model_validate(m) for m in body]
      else:
        updated_models = [public_model.model_validate(body)]

      return updated_models
    except ValidationError as e:
      logger.error(f"Error while loading {public_model} from the db: {e.errors(include_url=False)}")
  elif res.status_code == 500:
    st.error("Something went wrong on the server side, we apologize for the inconvenience")
  else:
    st.error(f"Something went wrong while updating information in the database: {res.status_code}")

def save_to_session_state(
  models: list[TPublic],
  session_state_field: str,
  ui_model: Type[BaseUI],
) -> None:
  """
  Helper function to save models to the session state.

  Args:
    models: A list of the models that we want to insert, these models usually come from the db response.
    session_state_field: The key used in order to insert the models in the session_state.
    This key must be created before hand.
    ui_model: The UI model that we'll validate the response models with.

  Raises:
    `ValidationError` if the UI model cannot validate the public models. 
  """
  for model in models:
    st.session_state[session_state_field][model.id] = ui_model.model_validate(model.model_dump())

def docx_to_pdf_bytes(docx_file: io.BytesIO | Path) -> io.BytesIO:
  """
  Helper function to convert a docx file to a pdf in order to display it in the UI with st.pdf

  Args:
    docx_file (io.BytesIO | Path): This can be either a io buffer or a pathlib.Path object.

  Returns:
    An in-memory buffer with the pdf bytes.
  """
  # Initialize empty buffer
  pdf_bytes = io.BytesIO()
  # Initialize empty path
  converted_pdf_path = ''
  # Extract bytes from the docx_file
  docx_bytes = docx_file.getvalue() if isinstance(docx_file, io.BytesIO) else docx_file.read_bytes()

  # Temporary DOCX input file
  with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp_docx:
    tmp_docx.write(docx_bytes)
    tmp_docx.flush()
    tmp_docx_path = tmp_docx.name

  # Temporary PDF output file
  with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
    tmp_pdf_path = tmp_pdf.name

  try:
    # LibreOffice requires output directory, not file path, so extract the directory
    output_dir = os.path.dirname(tmp_pdf_path)

    # Run libreoffice in the terminal.
    subprocess.run(
      [
        'libreoffice', '--headless', '--convert-to', 'pdf', '--outdir', output_dir, tmp_docx_path
      ],
      check=True
    )
    # Exctract the converted file path
    converted_pdf_path = os.path.join(
      output_dir,
      os.path.splitext(os.path.basename(tmp_docx_path))[0] + ".pdf"
    )

    # Read PDF into BytesIO
    with open(converted_pdf_path, "rb") as f:
      pdf_bytes.write(f.read())

  finally:
    # Clean up temp files
    os.unlink(tmp_docx_path)
    if os.path.exists(converted_pdf_path):
      os.unlink(converted_pdf_path)

  pdf_bytes.seek(0)
  return pdf_bytes

def scan_cv(cv: UploadedFile, username: str = USERNAME, url: str = URL) -> None:
  """
  Sends a POST request to the server in order to analyze and extract information from a given CV file.

  Args:
    cv: An UploadedFile instance to send to the backend, it must be a PDF file.
    username: The username used in order to authenticate the user in the backend, defaults to the USERNAME constant
    url: The base url where to send the request, defaults to the URL constant.

  Raises:
    `AttributeError`: If the given UploadedFile is not a pdf. 
  """
  if not cv.type == 'pdf':
    raise AttributeError(
      "The uplaoded file must be of pdf format."
    )
  final_url = f"{url}/from-file/{username}"
  payload = { "cv_file": (cv.name, cv.getvalue(), 'application/pdf'), }
  try:
    res = requests.post(final_url, files=payload)
  except requests.RequestException as e:
    st.error(f"Unable to send request to url: {final_url}, error: {str(e)}")
    return

  if res.status_code == 200:
    # When we post to this url, the backend saves the new information in the db.
    # We therefore only have to rerun the application in order to display the changes in the UI.
    st.rerun()
  elif res.status_code == 500:
    st.error("Something went wrong on the server side, we apologize for the inconvenience")
  else:
    st.error(f"Something went wrong while updating information in the database: {res.status_code}")

def start_customize_cv(
    model_creativity: float,
    model_tone: str,
    model_instructions: str | None,
    job_listing_url: str,
    username: str = USERNAME,
    url: str = URL
  ) -> bool:
  """
  Helper function that calls the main endpoint of the backend: The AI powered CV Customization.

  Args:
    model_creativty: Sets the internal model temperature. Useful in order to obtain more grounded results.
    model_tone: Instructs the models on the preferred tone of both the CV and CVL.
    model_instructions: Any extra instructions that one wants to pass to the model in his prompt.
    job_listing_url: The url that links to the desired job listing.
    username: The username used to authenticate the user on the server, defaults to the USERNAME constant.
    url: The base url where to send the request, defaults to the URL constant.

  Returns:
    A boolean flag to indicate if the process was successfull or not 
  """
  final_url = f"{url}/customize/{username}"

  payload = {
      "model_creativity_level": model_creativity,
      "model_tone": model_tone,
      "job_listing_url": job_listing_url,
      "model_instructions": model_instructions
  }
  return _customize_cv(payload, final_url)

def _customize_cv(payload: dict[str, Any], url: str) -> bool:
  """
  Internal, private function that actually handles the request process.
  The response is streamed from the backend as the CV customization process
  requires some time. In the chunks yielded from the backend we can find a 
  "task" field that can be used to display live updates in the UI.

  Args:
    payload: A python dictionary with all the required fields, I aim to make this a pydantic model instead.
    url: The base url where to send the request.

  THIS FUNCTION SHOULD NOT BE CALLED DIRECTLY AS IT IS INTERNAL AND SCOPED FOR DIFFERENT USES.
  """
  try:
    with requests.post(url, params=payload, stream=True) as res:
      if res.status_code == 200:
        for line in res.iter_lines(decode_unicode=True):
          if not line:
            continue

          data = json.loads(line)
          if "task" in data.keys():
            # Here we receive updates on the backend status
            with st.spinner(show_time=True):
              st.write(data['task'])
          elif "error" in data.keys():
            # Here we receive some server sent custom errors (such as invalid url)
            st.error(data['error'])
            return False
          else:
            # Here we have the final output to validate
            improved_cv_and_cvl = EnrichersCrewOutput.model_validate(data)
            st.session_state.cv = improved_cv_and_cvl.final_formatted_cv
            st.session_state.cvl = improved_cv_and_cvl.final_formatted_cvl
            return True
      else:
        # If the status_code is not of a success, then we set empty field and flash an error
        st.session_state.cv = None
        st.session_state.cvl = None
        st.error(f"Something went wrong on the server side; status code: {res.status_code}")
        return False

  except requests.RequestException as e:
    # If the request failes, then we set empty field and flash an error
    st.session_state.cv = None
    st.session_state.cvl = None
    st.error(f"Error while sending the request: {str(e)}")
    return False
  return True

@st.cache_resource
def populate_template(ctx: CVLOutput | CVOutput, input_path: str | Path) -> io.BytesIO:
  """
  Helper function to insert CV and CVL information inside of a docx template file.
  The function first modifes a docx file, and then converts it into a pdf, in order to
  display it in the UI with st.pdf
  
  Args:
    ctx: The context needed in order to insert information in the DocxTemplate.
    input_path (str | Path): The path that leads to the template file.

  Returns:
    An in-memory buffer with the pdf bytes.

  Raises:
    `TemplateError`: Since the models are validated already and the templated tested, the only reason why
    this error should be raisen is if the input file leads to a wrong template file.
    `FileNotFoundError`: If the file does not exist on the machine.
  """
  # Create empty buffer for the document file
  docx_buffer = io.BytesIO()
  # Load the template file
  template = DocxTemplate(input_path)
  
  try:
    template.render(ctx.model_dump())
  except TemplateError:
    raise TemplateError(
      f"Wrong ctx for the given input path, got ctx of type: {type(ctx)} and input path of: {input_path}"
    )
  # Save the modifies template in the buffer
  template.save(docx_buffer)
  # Go back to the beginning
  docx_buffer.seek(0)
  return docx_to_pdf_bytes(docx_buffer)

# --- MODELS --- #
class BaseUI(BaseModel, ABC):
  """
  Abstract base implementation for all of the UI models.
  It automatically generates Ids for the models as well as provide some helper methods and a
  stable framework in order to build them.
  """
  id: UUID = Field(default_factory=uuid4)

  @abstractmethod
  def display(self, method: Literal['create', 'update']) -> None:
    """
    Implementation to call in order to show a model in the UI.

    Args:
      method (`create` | `update`): Use create when creating a new model and update when modifying an
      existing one. Based on the value of the method, the actual implmentation will dispatch to the right method
      (either `update` or `save`) when the user submits the form in this display.
    """

  @abstractmethod
  def save(self) -> None:
    """
    Implementation to call as WidgetCallBack on the form_submit_button when creating a model
    """

  @abstractmethod
  def update(self) -> None:
    """
    Implementation to call as WidgetCallBack on the form_submit_button when updating an existing model
    """

  @abstractmethod
  def is_valid(self) -> bool:
    """
    Implementation to check if a form has been filled with all of the required fields.

    Returns:
      A boolean flag in order to determine if a models is valid or not.
    """

  def _to_create(self, create_model: Type[TCreate]) -> TCreate:
    """
    Base implementation in order to facilitate the conversion to a Create Model.
    This method loops in the model.__class__.model_computed_fields and accesses the ones that start with
    `_` and end with `key`.
    These fields will be used in order to extract values from the streamlit.session_state
    and used in order to create the CreateModel.
    Each one of these field might either have an alias that will be used to compute the fields 
    of the CreateModel or, in case that an alias is not assigned,
    the function will try to extract the original name of the field by removing the leading `_` and the trailing `_key`..
    In order to add the desired fields for each model, add `@computed_field` on the model property and assign it an alias.

    Args:
      create_model: The Pydantic Model Type to use in order to perfom data validation.

    Returns:
      The created and validated model if successfull.

    Raises:
      `AttributeError`: If a field does not have an alias and the function is not able to 
      deduct the original field from a property.
      `ValueError`: If a computed field does not start with `_` and end with `_key`.
      `ValidationError`: If the Create Model cannot be validated from the computed fields.
    """
    model_kwargs = {}
    for field_name, field_info in self.__class__.model_computed_fields.items():
      if not field_name.startswith('_') and not field_name.endswith('_key'):
        raise ValueError(
          "Index keys should start with a `_` and finish with `_key`"
        )
      # Try to extract the alias of the computed field if it exists
      if field_info.alias is not None:
        # Extract the value of the model's field from the session state
        model_kwargs[field_info.alias] = st.session_state.get(getattr(self, field_name))
        continue

      # Else wise we try to deduct the original model's field
      field = field_name.removeprefix('_').removesuffix('_key')
      if field not in self.__class__.model_fields:
        raise AttributeError(
          f"Cannot create the create model for {self.__class__.__name__} "
          f"since alias is missing and no mapping attribute has been found for computed_field: {field_name}"
          f"Model attributes: {self.__class__.__dict__}"
          f"Field: {field}"
        )
      # Extract the value of the model's field from the session state
      model_kwargs[field] = st.session_state.get(getattr(self, field_name))

    # Return the validated model
    return create_model(**model_kwargs)

  def error_msg(self, field: str) -> None:
      """
      Display an error message in the Streamlit UI.

      Args:
          field (str): The missing field from the parent object.
      """
      class_name = self.__class__.__name__.removesuffix("UI")
      st.error(f"Missing required field '{field}' in {class_name}")

# --- ACTUAL MODEL IMPLEMENTATIONS --- #
class PIIUI(BaseUI):
  email: str | None = None
  first_name: str | None = None
  last_name: str | None = None
  phone_number: str | None = None
  address: str | None = None
  linkedin_url: str | None = None
  github_url: str | None = None
  personal_website: str | None = None

  @computed_field
  @property
  def _email_key(self) -> str: return f"{self.id}_pii_email"
  @computed_field
  @property
  def _first_name_key(self) -> str: return f"{self.id}_pii_first_name"
  @computed_field
  @property
  def _last_name_key(self) -> str: return f"{self.id}_pii_last_name"
  @computed_field
  @property
  def _phone_number_key(self) -> str: return f"{self.id}_pii_phone"
  @computed_field
  @property
  def _address_key(self) -> str: return f"{self.id}_pii_address"
  @computed_field
  @property
  def _linkedin_url_key(self) -> str: return f"{self.id}_pii_linkedin"
  @computed_field
  @property
  def _github_url_key(self) -> str: return f"{self.id}_pii_gh_url"
  @computed_field
  @property
  def _personal_website_key(self) -> str: return f"{self.id}_pii_personal_url"

  def display(self, method: Literal['create', 'update']) -> None:
    with st.form(f"{self.id}_pii_form"):
      st.text_input("Email", self.email, key=self._email_key)
      col1, col2 = st.columns(2)
      with col1:
        st.text_input("First Name", self.first_name, key=self._first_name_key)
      with col2:
        st.text_input("Last Name", self.last_name, key=self._last_name_key)

      st.text_input("Phone Number", self.phone_number, key=self._phone_number_key)
      st.text_input("Address", self.address, key=self._address_key)
      st.text_input("LinkedIn URL", self.linkedin_url, key=self._linkedin_url_key)
      st.text_input("GitHub URL", self.github_url, key=self._github_url_key)
      st.text_input("Personal Website", self.personal_website, key=self._personal_website_key)

      st.form_submit_button("Save", width=1000, type='primary', on_click=self.save if method == 'create' else self.update)

  def save(self) -> None:
    if self.is_valid():
      public_models = save_to_db(
        "pii",
        user,
        [self._to_create(PIICreate)],
        PIIPublic
      )
      if public_models:
        save_to_session_state(public_models, 'pii', PIIUI)

  def update(self) -> None:
    if self.is_valid():
      new_models = update_in_db(
        'pii',
        [self._to_create(PIICreate)],
        [self.id],
        PIIPublic)
      if new_models:
        save_to_session_state(new_models, 'pii', PIIUI)

  def is_valid(self) -> bool:
    if not st.session_state.get(self._email_key): self.error_msg('email'); return False
    if not st.session_state.get(self._first_name_key): self.error_msg('first name'); return False
    if not st.session_state.get(self._last_name_key): self.error_msg('last name'); return False
    return True

class ExperienceUI(BaseUI):
  title: str | None = None
  company: str | None = None
  description: str | None = None
  start_date: datetime.datetime | datetime.date | None = None
  end_date: datetime.datetime | datetime.date | None = None
  remote: bool = False
  city: str | None = None
  country: str | None = None
  employment_type: str | None = None

  @computed_field
  @property
  def _title_key(self) -> str: return f"{self.id}_job_title"
  @computed_field
  @property
  def _company_key(self) -> str: return f"{self.id}_company"
  @computed_field
  @property
  def _description_key(self) -> str: return f"{self.id}_description"
  @computed_field
  @property
  def _start_date_key(self) -> str: return f"{self.id}_start_date"
  @computed_field
  @property
  def _end_date_key(self) -> str: return f"{self.id}_end_date"
  @computed_field
  @property
  def _remote_key(self) -> str: return f"{self.id}_remote"
  @computed_field
  @property
  def _city_key(self) -> str: return f"{self.id}_city"
  @computed_field
  @property
  def _country_key(self) -> str: return f"{self.id}_country"
  @computed_field
  @property
  def _employment_type_key(self) -> str: return f"{self.id}_employment_type"

  def display(self, method: Literal['create', 'update']):
    select_box_options = ["Full-Time", "Part-Time", "Internship"]

    with st.form(f"{self.id}_experience_form"):
      st.text_input("Job Title", self.title, key=self._title_key)
      st.text_input("Company Name", self.company, key=self._company_key)
      st.text_area("Job Description", self.description, key=self._description_key)
      st.date_input(
        "Start Date",
        value=self.start_date if self.start_date else "today",
        key=self._start_date_key
      )
      st.date_input(
        "End Date",
        self.end_date if self.end_date else 'today',
        key=self._end_date_key
      )
      remote = st.toggle("Remote", self.remote, key=self._remote_key)
      st.text_input("City", self.city, disabled=remote, key=self._city_key)
      st.text_input("Country", self.country, disabled=remote, key=self._country_key)
      st.selectbox(
        "Employment Type",
        select_box_options,
        select_box_options.index(self.employment_type) if self.employment_type else 0,
        key=self._employment_type_key
      )
      st.form_submit_button("Save", width=1000, type='primary', on_click=self.save if method == 'create' else self.update)

  def save(self) -> None:
    if self.is_valid():
      public_models = save_to_db(
        "work_experience", user, 
        [self._to_create(WorkExperienceCreate)],
        WorkExperiencePublic
      )
      if public_models:
        save_to_session_state(public_models, 'experiences', ExperienceUI)

  def update(self) -> None:
    if self.is_valid():
      updated_models = update_in_db(
        'work_experience',
        [self._to_create(WorkExperienceCreate)], [self.id], WorkExperiencePublic
      )
      if updated_models:
        save_to_session_state(updated_models, 'experiences', ExperienceUI)

  def is_valid(self) -> bool:
    if not st.session_state.get(self._title_key): self.error_msg("job title"); return False
    if not st.session_state.get(self._company_key): self.error_msg("company name"); return False
    if not st.session_state.get(self._description_key): self.error_msg("job description"); return False
    if not st.session_state.get(self._start_date_key): self.error_msg("start date"); return False
    if not st.session_state.get(self._end_date_key): self.error_msg("end date"); return False
    return True
  
class ProjectUI(BaseUI):
  title: str | None = None
  description: str | None = None
  technologies: str | None = None
  link: str | None = None

  @computed_field
  @property
  def _title_key(self) -> str: return f"{self.id}_project_title"
  @computed_field
  @property
  def _description_key(self) -> str: return f"{self.id}_project_description"
  @computed_field(alias='technologies')
  @property
  def _technologies_key(self) -> str: return f"{self.id}_project_technologies"
  @computed_field
  @property
  def _link_key(self) -> str: return f"{self.id}_project_link"

  def display(self, method: Literal['create', 'update']) -> None:
    with st.form(f"{self.id}_project_form"):
      st.text_input("Project Name", value=self.title, key=self._title_key)
      st.text_area("Description", value=self.description, key=self._description_key)
      st.text_area("Tools Used (comma-separated)", value=self.technologies, key=self._technologies_key)
      st.text_input("Project Link", value=self.link, key=self._link_key)
      st.form_submit_button("Save", width=1000, type='primary', on_click=self.save if method == 'create' else self.update)

  def save(self) -> None:
    if self.is_valid():
      public_models = save_to_db(
        'projects', user,
        [self._to_create(ProjectCreate)],
        ProjectPublic
      )
      if public_models:
        save_to_session_state(public_models, 'projects', ProjectUI)

  def update(self) -> None:
    if self.is_valid():
      updated_models = update_in_db(
        'projects',
        [self._to_create(ProjectCreate)], [self.id], ProjectPublic
      )
      if updated_models:
        save_to_session_state(updated_models, 'projects', ProjectUI)

  def is_valid(self) -> bool:
    if not st.session_state.get(self._title_key): self.error_msg("title"); return False
    if not st.session_state.get(self._description_key): self.error_msg("description"); return False
    return True
  
class CertificationUI(BaseUI):
  name: str | None = None
  issuer: str | None = None
  issue_date: datetime.datetime | datetime.date | None = None
  expiration_date: datetime.datetime | datetime.date | None = None
  credential_url: str | None = None
  expires: bool = False
  
  def model_post_init(self, context: Any) -> None:
    self.expires: bool = True if self.expiration_date else False

  @computed_field
  @property
  def _name_key(self) -> str: return f"{self.id}_name"
  @computed_field
  @property
  def _issuer_key(self) -> str: return f"{self.id}_issuer"
  @computed_field
  @property
  def _issue_date_key(self) -> str: return f"{self.id}_issue_date"
  @computed_field
  @property
  def _expiration_date_key(self) -> str: return f"{self.id}_exp_date"
  @computed_field
  @property
  def _credential_url_key(self) -> str: return f"{self.id}_url"
  @computed_field
  @property
  def _expires_key(self) -> str: return f"{self.id}_expires"

  def display(self, method: Literal['create'] | Literal['update']) -> None:
    with st.form(f"{self.id}_certification_form"):
      st.text_input("Certification Name", value=self.name, key=self._name_key)
      st.text_input("Issuer", value=self.issuer, key=self._issuer_key)
      _ = st.toggle("Has Expiration Date?", value=self.expires, key=self._expires_key)
      col1, col2 = st.columns(2)
      with col1: st.date_input(
        "Issue Date",
        value=self.issue_date if self.issue_date else 'today',
        key=self._issue_date_key
        )
      with col2: exp = st.date_input(
        "Expiration Date",
        value=self.expiration_date if self.expiration_date else 'today',
        key=self._expiration_date_key
        )
      st.text_input("Credential URL", value=self.credential_url, key=self._credential_url_key)
      st.form_submit_button("Save", width=1000, type='primary', on_click=self.save if method == 'create' else self.update)

  def save(self) -> None:
    if self.is_valid():
      public_models = save_to_db(
        'certifications', user,
        [self._to_create(CertificationCreate)],
        CertificationPublic
      )
      if public_models:
        save_to_session_state(public_models, 'certifications', CertificationUI)

  def update(self) -> None:
    if self.is_valid():
      updated_models = update_in_db(
        "certifications",
        [self._to_create(CertificationCreate)], [self.id], CertificationPublic
      )
      if updated_models:
        save_to_session_state(updated_models, 'certifications', CertificationUI)

  def is_valid(self) -> bool:
    if not st.session_state.get(self._name_key): self.error_msg("name"); return False
    if not st.session_state.get(self._issuer_key): self.error_msg("issuer"); return False
    if not st.session_state.get(self._issue_date_key): self.error_msg("issue date"); return False
    return True

class EducationUI(BaseUI):
  degree: str | None = None
  institution: str | None = None
  start_date: datetime.datetime | datetime.date | None = None
  end_date: datetime.datetime | datetime.date | None = None

  @computed_field
  @property
  def _degree_key(self) -> str: return f"{self.id}_degree"
  @computed_field
  @property
  def _institution_key(self) -> str: return f"{self.id}_institution"
  @computed_field
  @property
  def _start_date_key(self) -> str: return f"{self.id}_start_date"
  @computed_field
  @property
  def _end_date_key(self) -> str: return f"{self.id}_end_date"

  def display(self, method: Literal['create'] | Literal['update']) -> None:
    with st.form(f"{self.id}_education_form"):
      st.text_input("Degree", value=self.degree, key=self._degree_key)
      st.text_input("Institution", value=self.institution, key=self._institution_key)
      col1, col2 = st.columns(2)
      with col1: st.date_input(
        "Start Date",
        value=self.start_date if self.start_date else 'today',
        key=self._start_date_key
      )
      with col2: st.date_input(
        "End Date",
        value=self.end_date if self.end_date else 'today',
        key=self._end_date_key
      )
      st.form_submit_button("Save", width=1000, type='primary', on_click=self.save if method == 'create' else self.update)

  def save(self) -> None:
    if self.is_valid():
      public_models = save_to_db(
        'education', user,
        [self._to_create(EducationCreate)],
        EducationPublic
      )
      if public_models:
        save_to_session_state(public_models, 'educations', EducationUI)

  def update(self) -> None:
    if self.is_valid():
      updated_models = update_in_db(
        'education',
        [self._to_create(EducationCreate)], [self.id], EducationPublic
      )
      if updated_models:
        save_to_session_state(updated_models, 'educations', EducationUI)

  def is_valid(self) -> bool:
    if not st.session_state.get(self._degree_key): self.error_msg("degree"); return False
    if not st.session_state.get(self._institution_key): self.error_msg("institution"); return False
    return True

class LanguageUI(BaseUI):
  language: str | None = None
  proficiency_level: str | None = None

  @computed_field
  @property
  def _language_key(self) -> str: return f"{self.id}_language"
  @computed_field
  @property
  def _proficiency_level_key(self) -> str: return f"{self.id}_proficiency_level"

  @property
  def _proficiency_level_slider_options(self) -> list[str]: return ["Basic", "Intermediate", "Fluent", "Native"]

  def display(self, method: Literal['create', 'update']) -> None:
    with st.form(f"{self.id}_language_form"):
      st.text_input("Language", self.language, key=self._language_key)
      st.select_slider(
        "Proficiency Level",
        options=self._proficiency_level_slider_options,
        value=self.proficiency_level,
        key=self._proficiency_level_key
      )

      st.form_submit_button("Save", type="primary", width=1000, on_click=self.save if method == 'create' else self.update)

  def save(self) -> None:
    if self.is_valid():
      public_models = save_to_db(
        'languages', user,
        [self._to_create(LanguageCreate)],
        LanguagePublic
      )
      if public_models:
        save_to_session_state(public_models, 'languages', LanguageUI)

  def update(self) -> None:
    if self.is_valid():
      updated_models = update_in_db(
        "languages",
        [self._to_create(LanguageCreate)], [self.id], LanguagePublic
      )
      if updated_models:
        save_to_session_state(updated_models, 'languages', LanguageUI)

  def is_valid(self) -> bool:
    if not st.session_state.get(self._language_key): self.error_msg("language"); return False
    if not st.session_state.get(self._proficiency_level_key): self.error_msg("proficiency level"); return False
    return True
  
class SkillUI(BaseUI):
  skill_name: str | None = None
  experience_level: str | None = None
  category: str | None = None

  @computed_field
  @property
  def _skill_name_key(self) -> str: return f"{self.id}_skill_name"
  @computed_field
  @property
  def _experience_level_key(self) -> str: return f"{self.id}_experience_level"
  @computed_field
  @property
  def _category_key(self) -> str: return f"{self.id}_category"

  @property
  def _experience_slider_options(self): return ["Beginner", "Intermediate", "Professional"]
  @property
  def _category_slider_options(self): return ["Soft Skill", "Technical Skill"]

  def display(self, method: Literal['create'] | Literal['update']) -> None:
    with st.form(f"{self.id}_skill_form"):
      st.text_input("Skill Name", self.skill_name, key=self._skill_name_key)
      st.select_slider(
        "Experience Level",
        options=self._experience_slider_options,
        value=self.experience_level,
        key=self._experience_level_key
      )
      st.selectbox(
        "Category",
        self._category_slider_options,
        self._category_slider_options.index(self.category) if self.category else 0,
        key=self._category_key
      )
      st.form_submit_button("Save", width=1000, type='primary', on_click=self.save if method == 'create' else self.update)

  def save(self) -> None:
    if self.is_valid():
      public_models = save_to_db(
        'skills', user,
        [self._to_create(SkillCreate)],
        SkillPublic
      )
      if public_models:
        save_to_session_state(public_models, 'skills', SkillUI)

  def update(self) -> None:
    if self.is_valid():
      updated_models = update_in_db(
        'skills',
        [self._to_create(SkillCreate)], [self.id], SkillPublic
      )
      if updated_models:
        save_to_session_state(updated_models, 'skills', SkillUI)

  def is_valid(self) -> bool:
    if not st.session_state.get(self._skill_name_key): self.error_msg("name"); return False
    if not st.session_state.get(self._experience_level_key): self.error_msg("experience level"); return False
    return True

# --- SESSION STATE FIELDS --- #
# Here we initialize the session state's fields by retrieving entities from the db, if present.
if not 'pii' in st.session_state:
  st.session_state.pii = {}
  pii = get_from_db('pii', PIIPublic)
  if pii:
    pii = pii[0]  # Grab only the first PII from the response
    save_to_session_state([pii], 'pii', PIIUI)

if not 'experiences' in st.session_state:
  st.session_state.experiences = {}
  experiences = get_from_db('work_experience', WorkExperiencePublic)
  if experiences:
    save_to_session_state(experiences, 'experiences', ExperienceUI)

if not 'projects' in st.session_state:
  st.session_state.projects = {}
  projects = get_from_db('projects', ProjectPublic)
  if projects:
    save_to_session_state(projects, 'projects', ProjectUI)

if not 'certifications' in st.session_state:
  st.session_state.certifications = {}
  certs = get_from_db('certifications', CertificationPublic)
  if certs:
    save_to_session_state(certs, 'certifications', CertificationUI)

if not 'educations' in st.session_state:
  st.session_state.educations = {}
  edu = get_from_db('education', EducationPublic)
  if edu:
    save_to_session_state(edu, 'educations', EducationUI)

if not "languages" in st.session_state:
  st.session_state.languages = {}
  lang = get_from_db('languages', LanguagePublic)
  if lang:
    save_to_session_state(lang, 'languages', LanguageUI)

if not 'skills' in st.session_state:
  st.session_state.skills = {}
  skills = get_from_db('skills', SkillPublic)
  if skills:
    save_to_session_state(skills, 'skills', SkillUI)

if not "cv" in st.session_state:
  st.session_state.cv = None
# We populate a display CV file in case that we have a valid one in the session state
elif "cv" in st.session_state and isinstance(st.session_state.cv, CVOutput):
  st.session_state['display_cv_file'] = populate_template(st.session_state.cv, INPUT_CV_PATH)

if not "cvl" in st.session_state:
  st.session_state.cvl = None
# We populate a display CVL file in case that we have a valid one in the session state
elif "cvl" in st.session_state and isinstance(st.session_state.cvl, CVLOutput):
  st.session_state['display_cvl_file'] = populate_template(st.session_state.cvl, INPUT_CVL_PATH)

# --- MAIN APP ---
def main():
  cv_file = st.session_state.get('display_cv_file')
  cvl_file = st.session_state.get('display_cvl_file')
  col1, col2 = st.columns(2, gap='medium')

  with col1:
    model_tab, template_tab, personal_tab = st.tabs(["Model", "Templates", "Personal"])

    # --- MODEL TAB ---
    with model_tab:
      st.header("Model Preferences")
      with st.container(border=True):
        st.slider("Creativity", 0, 100, 70, key='model_creativity')
        st.text_input("Tone", placeholder="e.g. Professional", key='model_tone')
        st.text_area(
          "Extra Instructions",
          placeholder="Enter custom instruction for the model(if needed)",
          key="model_instructions"
        )

    # --- TEMPLATE TAB ---
    with template_tab:
      st.header("CV & Cover Letter")
      cv_template = st.selectbox("CV Template", ("Basic"), disabled=True)
      cvl_template = st.selectbox("Cover Letter Template", ("Basic"), disabled=True)
      left, center, right = st.columns((1, 3, 1)) 
      with center: st.subheader("More Templates coming soon!")

    # --- PERSONAL TAB ---
    with personal_tab:
      st.header("Personal Info")

      cv = st.file_uploader(
        "**_Extract Info from CV_**",
        type='pdf',
        key='cv_file'
      )
      if st.button("Scan CV", type='primary'):
        if not cv:
          st.error("Please select a file")
        else:
          scan_cv(cv)

      pii_tab, exp_tab, proj_tab, cert_tab, edu_tab, lang_tab, skills_tab = st.tabs(
        ["Private Info", "Experience", "Projects", "Certifications",
         "Education", "Languages", "Skills"],
      )

      with pii_tab:
        st.subheader("Private Information")

        if len(st.session_state.pii) > 0:
          for id, pii in st.session_state.pii.items():
            pii.display('update')
        else:
          pii = PIIUI()
          pii.display('create')

      with exp_tab:
        left, right = st.columns((3, 1))
        with left: st.subheader("Experience")
        with right: add_exp = st.button("Add Experience")
        new_exp = ExperienceUI()

        for id, exp in st.session_state.experiences.items():
          exp.display('update')

        if add_exp:
          new_exp.display("create")

      with proj_tab:
        left, right = st.columns((3, 1))
        with left: st.subheader("Projects")
        with right: add_proj = st.button("Add Project")
        new_proj = ProjectUI()

        for id, proj in st.session_state.projects.items():
          proj.display('update')

        if add_proj:
          new_proj.display("create")

      with cert_tab:
        left, right = st.columns((3, 1))
        with left: st.subheader("Certifications")
        with right: add_cert = st.button("Add Certification")
        new_cert = CertificationUI()

        for id, cert in st.session_state.certifications.items():
          cert.display('update')

        if add_cert:
          new_cert.display("create")

      with edu_tab:
        left, right = st.columns((3, 1))
        with left: st.subheader("Education")
        with right: add_edu = st.button("Add Education")
        new_edu = EducationUI()

        for id, edu in st.session_state.educations.items():
          edu.display('update')

        if add_edu:
          new_edu.display("create")

      with lang_tab:
        left, right = st.columns((3, 1))
        with left: st.subheader("Languages")
        with right: add_lang = st.button("Add Language")
        new_lang = LanguageUI()

        if add_lang:
          new_lang.display('create')

        for id, lang in st.session_state.languages.items():
          lang.display('update')

      with skills_tab:
        left, right = st.columns((3, 1))
        with left: st.subheader("Skills")
        with right: add_skill = st.button("Add Skill")
        new_skill = SkillUI()

        if add_skill:
          new_skill.display('create')

        for id, skill in st.session_state.skills.items():
          skill.display('update')

  with col2:
    with st.container(border=True, horizontal_alignment='center'):
      st.text("Insert job listing url:")
      with st.container(horizontal=True, vertical_alignment='bottom'):
        st.text_input("job_listing_input", key='job_listing_url', label_visibility='hidden')
        customize_btn =  st.button(label="Customize", type="primary")

    if customize_btn:
      with st.status("Customizing CV and CVL", expanded=True) as status:
        model_creativity_level: float = (
          st.session_state.get("model_creativity", 70) / 100
          if st.session_state.get("model_creativity", 70) != 0
          else 0
        )
        model_tone: str = st.session_state.get("model_tone", 'professional')
        job_listing_url: str | None = st.session_state.get("job_listing_url")
        model_instructions: str | None = st.session_state.get("model_instructions")
        if not job_listing_url:
          st.error("Please enter a valid job listing url.")
          status.update(state="error")
          st.stop()

        customized = start_customize_cv(model_creativity_level, model_tone, model_instructions, job_listing_url)
        if not customized:
          status.update(state="error")
          st.stop()

        st.session_state['display_cv_file'] = populate_template(st.session_state.cv, INPUT_CV_PATH).getvalue()
        st.session_state['display_cvl_file'] = populate_template(st.session_state.cvl, INPUT_CVL_PATH).getvalue()
        status.update(
          state='complete',
          expanded=False
        )

    file_display_selection = st.pills("file_display_selection", ['CV', 'CVL'], label_visibility='hidden')
    if file_display_selection == 'CV':
      if cv_file:
        st.pdf(cv_file, height=600)
        st.download_button("Download", data=st.session_state.display_cv_file, file_name="CustomCV.pdf")

    if file_display_selection == 'CVL':
      if cvl_file:
        st.pdf(cvl_file, height=600)
        st.download_button("Download", data=st.session_state.display_cvl_file, file_name='CustomCVL.pdf')

if __name__ == "__main__":
  main()
