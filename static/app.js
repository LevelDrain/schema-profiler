"use strict";

const $ = (selector) => document.querySelector(selector);
const uuid = () => crypto.randomUUID();
const shuffle = (items) => {
  const copy = [...items];
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
};
const escapeHtml = (value) => String(value)
  .replaceAll("&", "&amp;").replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;").replaceAll('"', "&quot;");

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: {"Content-Type": "application/json"},
    ...options,
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.error || `HTTP ${response.status}`);
  }
  return response.json();
}

const relationTrials = [
  {
    taskType: "relation_mapping",
    title: "関係対応",
    instruction: "左の要素を選び、同じ役割を持つ右の要素を選んで、3組の対応を作ってください。",
    source: {
      label: "研究チーム",
      nodes: [
        {id: "s_lead", label: "主任研究者"},
        {id: "s_member", label: "分析担当"},
        {id: "s_output", label: "報告書"},
      ],
      relations: ["主任研究者 → 分析担当：依頼する", "分析担当 → 報告書：作成する"],
    },
    target: {
      label: "料理店",
      nodes: [
        {id: "t_output", label: "料理"},
        {id: "t_lead", label: "料理長"},
        {id: "t_member", label: "調理担当"},
      ],
      relations: ["料理長 → 調理担当：指示する", "調理担当 → 料理：作成する"],
    },
    answer: {s_lead: "t_lead", s_member: "t_member", s_output: "t_output"},
  },
  {
    taskType: "relation_mapping",
    title: "関係対応",
    instruction: "外見や名称ではなく、関係の中で果たす役割を対応付けてください。",
    source: {
      label: "河川",
      nodes: [
        {id: "s_origin", label: "水源"},
        {id: "s_path", label: "川"},
        {id: "s_goal", label: "湖"},
      ],
      relations: ["水源 → 川：流れ込む", "川 → 湖：運ぶ"],
    },
    target: {
      label: "情報システム",
      nodes: [
        {id: "t_path", label: "通信路"},
        {id: "t_goal", label: "データベース"},
        {id: "t_origin", label: "センサー"},
      ],
      relations: ["センサー → 通信路：送信する", "通信路 → データベース：運ぶ"],
    },
    answer: {s_origin: "t_origin", s_path: "t_path", s_goal: "t_goal"},
  },
];

const similarityTrials = [
  {
    taskType: "structural_choice",
    title: "表面類似性と構造類似性",
    instruction: "基準と最も似た「関係構造」を持つ候補を1つ選んでください。",
    base: {title: "基準：赤い円の系", text: "大きな赤い円が小さな赤い円を囲み、小さな円が星を押す。"},
    candidates: [
      {id: "surface", title: "候補A", text: "大きな赤い円と小さな赤い円が、並んで星を見る。", kind: "surface"},
      {id: "structural", title: "候補B", text: "大きな青い四角が小さな青い四角を囲み、小さな四角が三角を押す。", kind: "structural"},
      {id: "other", title: "候補C", text: "三角が四角を囲み、円がその三角を押す。", kind: "other"},
    ],
    answer: "structural",
  },
  {
    taskType: "structural_choice",
    title: "表面類似性と構造類似性",
    instruction: "登場物の種類より、誰が誰に何をしているかを比較してください。",
    base: {title: "基準：学校", text: "先生が生徒に問題を渡し、生徒が答えを図書館へ届ける。"},
    candidates: [
      {id: "other", title: "候補A", text: "編集者が記事を保管し、記者が編集者へ写真を渡す。", kind: "other"},
      {id: "surface", title: "候補B", text: "先生が図書館から本を受け取り、生徒と一緒に読む。", kind: "surface"},
      {id: "structural", title: "候補C", text: "指令所が探査機へ命令を送り、探査機が観測結果を基地へ届ける。", kind: "structural"},
    ],
    answer: "structural",
  },
];

const reconstructionTrials = [
  {
    taskType: "structure_reconstruction",
    title: "構造再構成",
    instruction: "左の3要素を、右の役割へ配置し、参照構造と同じ関係にしてください。",
    reference: "参照：発信者 → 仲介者 → 受信者",
    items: [
      {id: "message_goal", label: "利用者"},
      {id: "message_origin", label: "アプリ"},
      {id: "message_path", label: "通知サービス"},
    ],
    roles: [
      {id: "origin", label: "発信者"},
      {id: "path", label: "仲介者"},
      {id: "goal", label: "受信者"},
    ],
    answer: {origin: "message_origin", path: "message_path", goal: "message_goal"},
  },
  {
    taskType: "structure_reconstruction",
    title: "構造再構成",
    instruction: "要素を選び、対応する役割をクリックしてください。配置済みの役割はクリックで解除できます。",
    reference: "参照：管理者が実行者を制御し、実行者が成果物を生成する",
    items: [
      {id: "factory_output", label: "製品"},
      {id: "factory_actor", label: "作業ロボット"},
      {id: "factory_control", label: "制御装置"},
    ],
    roles: [
      {id: "control", label: "管理者"},
      {id: "actor", label: "実行者"},
      {id: "output", label: "成果物"},
    ],
    answer: {control: "factory_control", actor: "factory_actor", output: "factory_output"},
  },
];

class BaseTask {
  constructor(trial, log) {
    this.trial = trial;
    this.log = log;
  }
  stimulus() {
    const copy = structuredClone(this.trial);
    delete copy.answer;
    return copy;
  }
}

class RelationMappingTask extends BaseTask {
  constructor(trial, log) {
    super(trial, log);
    this.selectedSource = null;
    this.mapping = {};
  }
  render(root, onChange) {
    const nodeButtons = (side, structure) => structure.nodes.map((node) =>
      `<button class="node" data-side="${side}" data-id="${node.id}">${escapeHtml(node.label)}</button>`
    ).join("");
    root.innerHTML = `
      <div class="mapping-layout">
        <div class="structure-box">
          <h3>${escapeHtml(this.trial.source.label)}</h3>
          <div class="node-grid">${nodeButtons("source", this.trial.source)}</div>
          <ul class="relation-list">${this.trial.source.relations.map((x) => `<li>${escapeHtml(x)}</li>`).join("")}</ul>
        </div>
        <div class="mapping-arrow">⇄</div>
        <div class="structure-box">
          <h3>${escapeHtml(this.trial.target.label)}</h3>
          <div class="node-grid">${nodeButtons("target", this.trial.target)}</div>
          <ul class="relation-list">${this.trial.target.relations.map((x) => `<li>${escapeHtml(x)}</li>`).join("")}</ul>
        </div>
      </div>
      <div class="mapping-list" id="mapping-list">まだ対応はありません。</div>`;
    root.addEventListener("click", (event) => {
      const button = event.target.closest(".node");
      if (!button) return;
      if (button.dataset.side === "source") {
        this.selectedSource = button.dataset.id;
        this.log("source_selected", {source_id: this.selectedSource});
      } else if (this.selectedSource) {
        const targetId = button.dataset.id;
        for (const [source, target] of Object.entries(this.mapping)) {
          if (target === targetId) delete this.mapping[source];
        }
        const previous = this.mapping[this.selectedSource] || null;
        this.mapping[this.selectedSource] = targetId;
        this.log("mapping_set", {
          source_id: this.selectedSource,
          target_id: targetId,
          previous_target_id: previous,
        });
        this.selectedSource = null;
      }
      this.update(root);
      onChange(this.isComplete());
    });
    root.querySelector("#mapping-list").addEventListener("click", (event) => {
      const remove = event.target.closest("[data-remove]");
      if (!remove) return;
      const sourceId = remove.dataset.remove;
      const targetId = this.mapping[sourceId];
      delete this.mapping[sourceId];
      this.log("mapping_removed", {source_id: sourceId, target_id: targetId});
      this.update(root);
      onChange(this.isComplete());
    });
    this.update(root);
  }
  update(root) {
    root.querySelectorAll(".node").forEach((button) => {
      const isSource = button.dataset.side === "source";
      button.classList.toggle("selected", isSource && button.dataset.id === this.selectedSource);
      const mapped = isSource
        ? Object.hasOwn(this.mapping, button.dataset.id)
        : Object.values(this.mapping).includes(button.dataset.id);
      button.classList.toggle("mapped", mapped);
    });
    const labels = Object.fromEntries(
      [...this.trial.source.nodes, ...this.trial.target.nodes].map((node) => [node.id, node.label])
    );
    const entries = Object.entries(this.mapping);
    root.querySelector("#mapping-list").innerHTML = entries.length
      ? entries.map(([source, target]) =>
        `<span class="mapping-chip">${escapeHtml(labels[source])} → ${escapeHtml(labels[target])}<button data-remove="${source}" aria-label="削除">×</button></span>`
      ).join("")
      : "まだ対応はありません。";
  }
  isComplete() {
    return Object.keys(this.mapping).length === this.trial.source.nodes.length;
  }
  result() {
    const correct = Object.entries(this.trial.answer)
      .filter(([source, target]) => this.mapping[source] === target).length;
    return {
      response: {mapping: this.mapping},
      metrics: {
        score: correct / Object.keys(this.trial.answer).length,
        correct_mappings: correct,
        total_mappings: Object.keys(this.trial.answer).length,
      },
    };
  }
}

class StructuralChoiceTask extends BaseTask {
  constructor(trial, log) {
    super(trial, log);
    this.selected = null;
    this.candidates = shuffle(trial.candidates);
  }
  stimulus() {
    return {...super.stimulus(), candidates: this.candidates};
  }
  render(root, onChange) {
    root.innerHTML = `
      <div class="base-card">
        <strong>${escapeHtml(this.trial.base.title)}</strong>
        <div class="mini-structure">${escapeHtml(this.trial.base.text)}</div>
      </div>
      <div class="candidate-grid">
        ${this.candidates.map((candidate) => `
          <button class="candidate" data-id="${candidate.id}">
            <strong>${escapeHtml(candidate.title)}</strong>
            <div class="mini-structure">${escapeHtml(candidate.text)}</div>
          </button>`).join("")}
      </div>`;
    root.addEventListener("click", (event) => {
      const candidate = event.target.closest(".candidate");
      if (!candidate) return;
      const previous = this.selected;
      this.selected = candidate.dataset.id;
      this.log(previous ? "choice_changed" : "choice_selected", {
        choice_id: this.selected,
        previous_choice_id: previous,
      });
      root.querySelectorAll(".candidate").forEach((node) =>
        node.classList.toggle("selected", node.dataset.id === this.selected)
      );
      onChange(true);
    });
  }
  isComplete() {
    return this.selected !== null;
  }
  result() {
    const selectedCandidate = this.trial.candidates.find(
      (candidate) => candidate.id === this.selected
    );
    return {
      response: {choice_id: this.selected, choice_kind: selectedCandidate.kind},
      metrics: {
        score: this.selected === this.trial.answer ? 1 : 0,
        selected_structural: selectedCandidate.kind === "structural",
        selected_surface: selectedCandidate.kind === "surface",
      },
    };
  }
}

class ReconstructionTask extends BaseTask {
  constructor(trial, log) {
    super(trial, log);
    this.selectedItem = null;
    this.placements = {};
  }
  render(root, onChange) {
    root.innerHTML = `
      <div class="base-card"><strong>${escapeHtml(this.trial.reference)}</strong></div>
      <div class="rebuild-layout">
        <div>
          <h3>要素</h3>
          <div class="item-pool">${this.trial.items.map((item) =>
            `<button class="movable" data-item="${item.id}">${escapeHtml(item.label)}</button>`
          ).join("")}</div>
        </div>
        <div>
          <h3>役割</h3>
          <div class="role-list">${this.trial.roles.map((role) =>
            `<button class="role" data-role="${role.id}"><strong>${escapeHtml(role.label)}</strong><span></span></button>`
          ).join("")}</div>
        </div>
      </div>`;
    root.addEventListener("click", (event) => {
      const item = event.target.closest("[data-item]");
      const role = event.target.closest("[data-role]");
      if (item && !item.classList.contains("used")) {
        this.selectedItem = item.dataset.item;
        this.log("item_selected", {item_id: this.selectedItem});
      } else if (role && this.selectedItem) {
        for (const [roleId, itemId] of Object.entries(this.placements)) {
          if (itemId === this.selectedItem) delete this.placements[roleId];
        }
        const previous = this.placements[role.dataset.role] || null;
        this.placements[role.dataset.role] = this.selectedItem;
        this.log("item_placed", {
          item_id: this.selectedItem,
          role_id: role.dataset.role,
          replaced_item_id: previous,
        });
        this.selectedItem = null;
      } else if (role && this.placements[role.dataset.role]) {
        const removed = this.placements[role.dataset.role];
        delete this.placements[role.dataset.role];
        this.log("placement_removed", {
          item_id: removed,
          role_id: role.dataset.role,
        });
      }
      this.update(root);
      onChange(this.isComplete());
    });
    this.update(root);
  }
  update(root) {
    const itemLabels = Object.fromEntries(
      this.trial.items.map((item) => [item.id, item.label])
    );
    root.querySelectorAll("[data-item]").forEach((button) => {
      button.classList.toggle("selected", button.dataset.item === this.selectedItem);
      button.classList.toggle(
        "used",
        Object.values(this.placements).includes(button.dataset.item)
      );
    });
    root.querySelectorAll("[data-role]").forEach((button) => {
      const itemId = this.placements[button.dataset.role];
      button.classList.toggle("filled", Boolean(itemId));
      button.querySelector("span").textContent = itemId
        ? `：${itemLabels[itemId]}`
        : "：未配置";
    });
  }
  isComplete() {
    return Object.keys(this.placements).length === this.trial.roles.length;
  }
  result() {
    const correct = Object.entries(this.trial.answer)
      .filter(([role, item]) => this.placements[role] === item).length;
    return {
      response: {placements: this.placements},
      metrics: {
        score: correct / Object.keys(this.trial.answer).length,
        correct_placements: correct,
        total_placements: Object.keys(this.trial.answer).length,
      },
    };
  }
}

const taskConstructors = {
  relation_mapping: RelationMappingTask,
  structural_choice: StructuralChoiceTask,
  structure_reconstruction: ReconstructionTask,
};

class ExperimentRunner {
  constructor() {
    this.sessionId = null;
    this.queue = [];
    this.index = 0;
    this.results = [];
    this.current = null;
    this.events = [];
    this.startedAtEpoch = null;
    this.startedAtPerformance = null;
    this.boundVisibility = () => {
      this.log(document.hidden ? "focus_left" : "focus_returned", {});
    };
  }

  async start(participantId) {
    const session = await api("/api/sessions", {
      method: "POST",
      body: JSON.stringify({participant_id: participantId}),
    });
    this.sessionId = session.session_id;
    const groups = [relationTrials, similarityTrials, reconstructionTrials];
    this.queue = shuffle(groups).flatMap((group) => shuffle(group));
    this.index = 0;
    this.results = [];
    $("#start-screen").classList.add("hidden");
    $("#experiment-screen").classList.remove("hidden");
    document.addEventListener("visibilitychange", this.boundVisibility);
    this.showTrial();
  }

  showTrial() {
    const trial = this.queue[this.index];
    this.events = [];
    this.startedAtEpoch = Date.now() / 1000;
    this.startedAtPerformance = performance.now();
    const Constructor = taskConstructors[trial.taskType];
    this.current = new Constructor(trial, (type, payload) => this.log(type, payload));

    $("#task-label").textContent = trial.title;
    $("#progress-label").textContent = `${this.index + 1} / ${this.queue.length}`;
    $("#progress-bar").style.width = `${(this.index / this.queue.length) * 100}%`;
    $("#trial-title").textContent = trial.title;
    $("#trial-instruction").textContent = trial.instruction;
    $("#confidence").value = "3";
    $("#confidence-value").textContent = "3";
    $("#submit-button").disabled = true;
    $("#trial-error").textContent = "";
    const previousRoot = $("#task-root");
    const root = previousRoot.cloneNode(false);
    previousRoot.replaceWith(root);
    this.current.render(root, (complete) => {
      $("#submit-button").disabled = !complete;
    });
    this.log("trial_presented", {task_type: trial.taskType});
  }

  log(eventType, payload) {
    if (this.startedAtPerformance === null) return;
    this.events.push({
      elapsed_ms: performance.now() - this.startedAtPerformance,
      event_type: eventType,
      payload,
    });
  }

  async submit() {
    if (!this.current.isComplete()) return;
    $("#submit-button").disabled = true;
    const completedAt = Date.now() / 1000;
    const duration = performance.now() - this.startedAtPerformance;
    const confidence = Number($("#confidence").value);
    const output = this.current.result();
    this.log("trial_submitted", {confidence, score: output.metrics.score});
    const record = {
      id: uuid(),
      session_id: this.sessionId,
      task_type: this.queue[this.index].taskType,
      trial_index: this.index,
      started_at: this.startedAtEpoch,
      completed_at: completedAt,
      duration_ms: duration,
      confidence,
      stimulus: this.current.stimulus(),
      response: output.response,
      metrics: output.metrics,
      events: this.events,
    };
    try {
      await api("/api/trials", {
        method: "POST",
        body: JSON.stringify(record),
      });
      this.results.push(record);
      this.index += 1;
      if (this.index < this.queue.length) {
        this.showTrial();
      } else {
        await this.finish();
      }
    } catch (error) {
      $("#trial-error").textContent = `保存できませんでした: ${error.message}`;
      $("#submit-button").disabled = false;
    }
  }

  async finish() {
    await api("/api/sessions/complete", {
      method: "POST",
      body: JSON.stringify({session_id: this.sessionId}),
    });
    document.removeEventListener("visibilitychange", this.boundVisibility);
    $("#experiment-screen").classList.add("hidden");
    $("#result-screen").classList.remove("hidden");
    this.renderSummary();
  }

  renderSummary() {
    const labels = {
      relation_mapping: "関係対応",
      structural_choice: "構造選択",
      structure_reconstruction: "構造再構成",
    };
    const grouped = Object.keys(labels).map((type) => {
      const rows = this.results.filter((result) => result.task_type === type);
      const score = rows.reduce((sum, row) => sum + row.metrics.score, 0) / rows.length;
      const duration = rows.reduce((sum, row) => sum + row.duration_ms, 0) / rows.length;
      return {type, label: labels[type], score, duration};
    });
    $("#summary-cards").innerHTML = grouped.map((group) => `
      <div class="summary-card">
        ${group.label}
        <strong>${Math.round(group.score * 100)}%</strong>
        <small>平均 ${Math.round(group.duration / 100) / 10} 秒</small>
      </div>`).join("");
    $("#trial-summary").innerHTML = `
      <table>
        <thead><tr><th>試行</th><th>課題</th><th>一致率</th><th>時間</th><th>確信度</th></tr></thead>
        <tbody>${this.results.map((result, index) => `
          <tr>
            <td>${index + 1}</td>
            <td>${labels[result.task_type]}</td>
            <td>${Math.round(result.metrics.score * 100)}%</td>
            <td>${Math.round(result.duration_ms / 100) / 10}秒</td>
            <td>${result.confidence}/5</td>
          </tr>`).join("")}</tbody>
      </table>`;
  }

  async download() {
    const data = await api(
      `/api/export?session_id=${encodeURIComponent(this.sessionId)}`
    );
    const blob = new Blob(
      [JSON.stringify(data, null, 2)],
      {type: "application/json"}
    );
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `structure-mapping-${this.sessionId}.json`;
    link.click();
    URL.revokeObjectURL(url);
  }
}

const runner = new ExperimentRunner();

$("#start-button").addEventListener("click", async () => {
  const participantId = $("#participant-id").value.trim();
  if (!participantId) {
    $("#start-error").textContent = "参加者IDを入力してください。";
    return;
  }
  $("#start-button").disabled = true;
  $("#start-error").textContent = "";
  try {
    await runner.start(participantId);
  } catch (error) {
    $("#start-error").textContent = `開始できませんでした: ${error.message}`;
    $("#start-button").disabled = false;
  }
});
$("#submit-button").addEventListener("click", () => runner.submit());
$("#confidence").addEventListener("input", (event) => {
  $("#confidence-value").textContent = event.target.value;
  runner.log("confidence_changed", {value: Number(event.target.value)});
});
$("#download-button").addEventListener("click", () => runner.download());
$("#restart-button").addEventListener("click", () => location.reload());
