extends Control

@onready var output_label: RichTextLabel = $VBoxContainer/OutputLabel
@onready var prompt_input: LineEdit = $VBoxContainer/HBoxContainer/PromptInput
@onready var send_button: Button = $VBoxContainer/HBoxContainer/SendButton
@onready var model_path_input: LineEdit = $VBoxContainer/ModelPathInput
@onready var open_button: Button = $VBoxContainer/OpenButton
@onready var status_label: Label = $VBoxContainer/StatusLabel

var session: LlamaSession
var current_request_id: int = -1

func _ready() -> void:
	session = LlamaSession.new()
	session.opened.connect(_on_opened)
	session.token_emitted.connect(_on_token_emitted)
	session.completed.connect(_on_completed)
	session.failed.connect(_on_failed)
	session.cancelled.connect(_on_cancelled)

	send_button.pressed.connect(_on_send_pressed)
	open_button.pressed.connect(_on_open_pressed)
	send_button.disabled = true
	status_label.text = "Status: Not loaded"

func _process(_delta: float) -> void:
	if session:
		session.poll()

func _on_open_pressed() -> void:
	var config = LlamaModelConfig.new()
	config.model_path = model_path_input.text
	config.n_ctx = 2048
	config.n_threads = -1

	status_label.text = "Status: Loading model..."
	var err = session.open(config)
	if err != 0:
		status_label.text = "Status: Failed to open (error %d)" % err

func _on_opened() -> void:
	status_label.text = "Status: Model loaded"
	send_button.disabled = false

func _on_send_pressed() -> void:
	if not session.is_open():
		return
	var prompt = prompt_input.text
	if prompt.is_empty():
		return
	output_label.text = ""
	var options = {"max_tokens": 256, "temperature": 0.8}
	current_request_id = session.generate_async(prompt, options)
	status_label.text = "Status: Generating..."
	send_button.disabled = true

func _on_token_emitted(request_id: int, token_text: String, _token_id: int) -> void:
	if request_id == current_request_id:
		output_label.text += token_text

func _on_completed(request_id: int, _text: String, stats: Dictionary) -> void:
	if request_id == current_request_id:
		var tps = stats.get("tokens_per_second", 0.0)
		status_label.text = "Status: Done (%.1f tok/s)" % tps
		send_button.disabled = false

func _on_failed(request_id: int, error_code: int, error_message: String, _details: String) -> void:
	if request_id == current_request_id:
		status_label.text = "Status: Error %d - %s" % [error_code, error_message]
		send_button.disabled = false

func _on_cancelled(request_id: int) -> void:
	if request_id == current_request_id:
		status_label.text = "Status: Cancelled"
		send_button.disabled = false

func _exit_tree() -> void:
	if session:
		session.close()
