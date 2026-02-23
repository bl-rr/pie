use crate::api::core::Queue;
use crate::api::inferlet;
use crate::instance::InstanceState;
use crate::model;
use crate::model::request::{ClassifyBatchRequest, ClassifyBatchResponse, Request};
use anyhow::Result;
use tokio::sync::oneshot;
use wasmtime::component::Resource;
use wasmtime_wasi::p2::{DynPollable, Pollable, subscribe};
use wasmtime_wasi::{WasiView, async_trait};

/// Re-export WIT-generated types for classify interface.
type WitClassificationInput = inferlet::core::classify::ClassificationInput;
type WitClassificationOutput = inferlet::core::classify::ClassificationOutput;

/// Async result holder for batch classification.
#[derive(Debug)]
pub struct ClassificationResult {
    receiver: oneshot::Receiver<ClassifyBatchResponse>,
    result: Option<Vec<WitClassificationOutput>>,
    done: bool,
}

#[async_trait]
impl Pollable for ClassificationResult {
    async fn ready(&mut self) {
        if self.done {
            return;
        }
        let resp = (&mut self.receiver).await.unwrap();
        self.result = Some(
            resp.results
                .into_iter()
                .map(|r| WitClassificationOutput {
                    label: r.label,
                    score: r.score,
                })
                .collect(),
        );
        self.done = true;
    }
}

impl inferlet::core::classify::Host for InstanceState {
    async fn classify_batch(
        &mut self,
        queue: Resource<Queue>,
        inputs: Vec<WitClassificationInput>,
    ) -> Result<Resource<ClassificationResult>> {
        let (svc_id, queue_id, priority) = self.read_queue(&queue)?;
        let (tx, rx) = oneshot::channel();

        let pairs: Vec<(String, String)> = inputs
            .into_iter()
            .map(|i| (i.premise, i.hypothesis))
            .collect();

        let req = Request::ClassifyBatch(ClassifyBatchRequest { pairs }, tx);
        model::submit_request(svc_id, queue_id, priority, req)?;

        let result = ClassificationResult {
            receiver: rx,
            result: None,
            done: false,
        };

        Ok(self.ctx().table.push(result)?)
    }
}

impl inferlet::core::classify::HostClassificationResult for InstanceState {
    async fn pollable(
        &mut self,
        this: Resource<ClassificationResult>,
    ) -> Result<Resource<DynPollable>> {
        subscribe(self.ctx().table, this)
    }

    async fn get(
        &mut self,
        this: Resource<ClassificationResult>,
    ) -> Result<Option<Vec<WitClassificationOutput>>> {
        let result = self.ctx().table.get_mut(&this)?;
        if result.done {
            Ok(result.result.clone())
        } else {
            Ok(None)
        }
    }

    async fn drop(&mut self, this: Resource<ClassificationResult>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}
