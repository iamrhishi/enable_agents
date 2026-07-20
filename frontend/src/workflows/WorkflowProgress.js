import React from 'react';
import './WorkflowProgress.css';

function WorkflowProgress({ instance }) {
  if (!instance || !instance.templateId) {
    return null;
  }

  // Build stages from instance data
  // We need template stages but we may not have them directly
  // For now, show based on what we know from instance
  const stages = [];
  const stageStates = instance.stageStates || {};
  const totalStages = instance.totalStages || 0;
  const currentIndex = instance.currentStageIndex || 0;

  // If we have currentStage info, we can infer some structure
  // But ideally we'd fetch full template or have it embedded
  for (let i = 0; i < totalStages; i++) {
    const isCompleted = i < currentIndex;
    const isCurrent = i === currentIndex && instance.status === 'running';
    const isPending = i > currentIndex;

    stages.push({
      index: i,
      name: `Stage ${i + 1}`,
      status: isCompleted ? 'completed' : isCurrent ? 'current' : 'pending',
    });
  }

  // If we have stage states with more info, use it
  const stageIds = Object.keys(stageStates);
  stageIds.forEach((stageId, idx) => {
    if (stages[idx]) {
      stages[idx].name = stageId.replace(/_/g, ' ').replace(/-/g, ' ');
      stages[idx].completedAt = stageStates[stageId].completedAt;
    }
  });

  // Override with currentStage info if available
  if (instance.currentStage && stages[currentIndex]) {
    stages[currentIndex].name = instance.currentStage.name;
  }

  return (
    <div className="workflow-progress">
      <div className="progress-header">
        <h3>Progress</h3>
        <span className="progress-fraction">
          {currentIndex} / {totalStages} completed
        </span>
      </div>

      <div className="progress-bar-container">
        <div className="progress-bar-track">
          <div
            className="progress-bar-fill"
            style={{ width: `${totalStages > 0 ? (currentIndex / totalStages) * 100 : 0}%` }}
          />
        </div>
      </div>

      <div className="stages-timeline">
        {stages.map((stage, idx) => (
          <div key={idx} className={`stage-item stage-${stage.status}`}>
            <div className="stage-indicator">
              {stage.status === 'completed' ? (
                <span className="indicator-check">✓</span>
              ) : stage.status === 'current' ? (
                <span className="indicator-current">{idx + 1}</span>
              ) : (
                <span className="indicator-pending">{idx + 1}</span>
              )}
            </div>
            <div className="stage-info">
              <span className="stage-name">{stage.name}</span>
              {stage.completedAt && (
                <span className="stage-time">
                  {new Date(stage.completedAt).toLocaleDateString()}
                </span>
              )}
            </div>
            {idx < stages.length - 1 && <div className="stage-connector" />}
          </div>
        ))}
      </div>
    </div>
  );
}

export default WorkflowProgress;
