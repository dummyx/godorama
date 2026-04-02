extends Control

@onready var status_label: Label = $VBox/StatusLabel
@onready var generation_model_input: LineEdit = $VBox/Paths/Grid/GenerationModelInput
@onready var embedding_model_input: LineEdit = $VBox/Paths/Grid/EmbeddingModelInput
@onready var open_button: Button = $VBox/Controls/OpenButton
@onready var ingest_button: Button = $VBox/Controls/IngestButton
@onready var question_input: LineEdit = $VBox/QuestionBox/QuestionInput
@onready var ask_button: Button = $VBox/QuestionBox/AskButton
@onready var cancel_button: Button = $VBox/QuestionBox/CancelButton
@onready var hits_label: RichTextLabel = $VBox/Panes/HitsPanel/HitsLabel
@onready var answer_label: RichTextLabel = $VBox/Panes/AnswerPanel/AnswerLabel

var corpus: RagCorpus
var answer_session: RagAnswerSession
var current_retrieve_request_id: int = -1
var current_answer_request_id: int = -1
var current_ingest_jobs: Array[int] = []

func _ready() -> void:
	corpus = RagCorpus.new()
	answer_session = RagAnswerSession.new()

	corpus.ingest_progress.connect(_on_ingest_progress)
	corpus.ingest_completed.connect(_on_ingest_completed)
	corpus.retrieve_completed.connect(_on_retrieve_completed)
	corpus.failed.connect(_on_corpus_failed)

	answer_session.token_emitted.connect(_on_token_emitted)
	answer_session.completed.connect(_on_answer_completed)
	answer_session.failed.connect(_on_answer_failed)
	answer_session.cancelled.connect(_on_answer_cancelled)

	open_button.pressed.connect(_on_open_pressed)
	ingest_button.pressed.connect(_on_ingest_pressed)
	ask_button.pressed.connect(_on_ask_pressed)
	cancel_button.pressed.connect(_on_cancel_pressed)

	ingest_button.disabled = true
	ask_button.disabled = true
	cancel_button.disabled = true
	status_label.text = "Status: Closed"

func _process(_delta: float) -> void:
	if corpus:
		corpus.poll()
	if answer_session:
		answer_session.poll()

func _on_open_pressed() -> void:
	var generation_config := LlamaModelConfig.new()
	generation_config.model_path = generation_model_input.text
	generation_config.n_ctx = 2048
	generation_config.n_threads = -1

	var corpus_config := RagCorpusConfig.new()
	corpus_config.storage_path = ProjectSettings.globalize_path("user://rag_demo.sqlite3")
	corpus_config.embedding_model_path = embedding_model_input.text
	corpus_config.embedding_n_ctx = 1024
	corpus_config.embedding_n_threads = -1
	corpus_config.parser_mode = "markdown"

	status_label.text = "Status: Opening generation and corpus models..."

	var answer_err := answer_session.open_generation(generation_config)
	if answer_err != 0:
		status_label.text = "Status: Failed to open generation model (%d)" % answer_err
		return

	var corpus_err := corpus.open(corpus_config)
	if corpus_err != 0:
		status_label.text = "Status: Failed to open corpus (%d)" % corpus_err
		answer_session.close_generation()
		return

	status_label.text = "Status: Ready. Ingest the demo fixtures."
	ingest_button.disabled = false
	ask_button.disabled = false

func _on_ingest_pressed() -> void:
	if not corpus.is_open():
		return

	current_ingest_jobs.clear()
	var fixture_paths := [
		"res://rag_fixtures/godot.txt",
		"res://rag_fixtures/sqlite.md",
		"res://rag_fixtures/llama.txt",
	]

	for fixture_path in fixture_paths:
		var absolute_path := ProjectSettings.globalize_path(fixture_path)
		var job_id := corpus.upsert_file_async(absolute_path, {"fixture": fixture_path})
		if job_id > 0:
			current_ingest_jobs.append(job_id)

	status_label.text = "Status: Ingesting %d fixture files..." % current_ingest_jobs.size()
	hits_label.text = ""
	answer_label.text = ""

func _on_ask_pressed() -> void:
	if not corpus.is_open() or not answer_session.is_generation_open():
		return
	if question_input.text.is_empty():
		return

	var retrieval_options := {
		"top_k": 3,
		"candidate_k": 6,
		"max_context_chunks": 3,
		"max_context_tokens": 512,
		"use_mmr": true,
	}
	var generation_options := {
		"max_tokens": 192,
		"temperature": 0.2,
	}

	hits_label.text = ""
	answer_label.text = ""
	current_retrieve_request_id = corpus.retrieve_async(question_input.text, retrieval_options)
	current_answer_request_id = answer_session.answer_async(corpus, question_input.text, retrieval_options, generation_options)
	cancel_button.disabled = false
	status_label.text = "Status: Retrieving and generating..."

func _on_cancel_pressed() -> void:
	if current_retrieve_request_id > 0:
		corpus.cancel_job(current_retrieve_request_id)
	if current_answer_request_id > 0:
		answer_session.cancel(current_answer_request_id)
	status_label.text = "Status: Cancelling..."

func _on_ingest_progress(job_id: int, done: int, total: int) -> void:
	if current_ingest_jobs.has(job_id):
		status_label.text = "Status: Ingesting %d / %d chunks for job %d..." % [done, total, job_id]

func _on_ingest_completed(job_id: int, stats: Dictionary) -> void:
	if current_ingest_jobs.has(job_id):
		status_label.text = "Status: Ingested %s (%d chunks)" % [stats.get("source_id", ""), stats.get("chunks_written", 0)]

func _on_retrieve_completed(request_id: int, hits: Array, _stats: Dictionary) -> void:
	if request_id != current_retrieve_request_id:
		return

	var lines: Array[String] = []
	for hit in hits:
		lines.append("[b]%s[/b] (%s:%s-%s)\n%s" % [
			hit.get("source_id", ""),
			hit.get("chunk_id", ""),
			hit.get("byte_start", 0),
			hit.get("byte_end", 0),
			hit.get("excerpt", ""),
		])
	hits_label.text = "\n\n".join(lines)

func _on_token_emitted(request_id: int, token_text: String, _token_id: int) -> void:
	if request_id == current_answer_request_id:
		answer_label.text += token_text

func _on_answer_completed(request_id: int, _text: String, citations: Array, stats: Dictionary) -> void:
	if request_id != current_answer_request_id:
		return

	cancel_button.disabled = true
	var citation_lines: Array[String] = []
	for citation in citations:
		citation_lines.append("- %s:%s-%s" % [
			citation.get("source_id", ""),
			citation.get("byte_start", 0),
			citation.get("byte_end", 0),
		])
	if not citation_lines.is_empty():
		answer_label.text += "\n\nCitations:\n" + "\n".join(citation_lines)
	status_label.text = "Status: Answer complete (%d packed chunks)" % stats.get("packed_chunks", 0)

func _on_corpus_failed(id_value: int, error_code: int, error_message: String, details: String) -> void:
	status_label.text = "Status: Corpus error %d - %s" % [error_code, error_message]
	if id_value == current_retrieve_request_id:
		hits_label.text = "[b]Retrieval failed[/b]\n%s\n%s" % [error_message, details]

func _on_answer_failed(request_id: int, error_code: int, error_message: String, details: String) -> void:
	if request_id != current_answer_request_id:
		return
	cancel_button.disabled = true
	status_label.text = "Status: Answer error %d - %s" % [error_code, error_message]
	answer_label.text = "[b]Generation failed[/b]\n%s\n%s" % [error_message, details]

func _on_answer_cancelled(request_id: int) -> void:
	if request_id == current_answer_request_id:
		cancel_button.disabled = true
		status_label.text = "Status: Answer cancelled"

func _exit_tree() -> void:
	if answer_session:
		answer_session.close_generation()
	if corpus:
		corpus.close()
