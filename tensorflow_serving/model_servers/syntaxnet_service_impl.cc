/* Copyright 2018 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "grpc/grpc.h"

#include "tensorflow_serving/model_servers/syntaxnet_service_impl.h"
#include "tensorflow_serving/model_servers/grpc_status_util.h"
#include "tensorflow_serving/servables/tensorflow/classification_service.h"
#include "tensorflow_serving/servables/tensorflow/get_model_metadata_impl.h"
#include "tensorflow_serving/servables/tensorflow/multi_inference_helper.h"
#include "tensorflow_serving/servables/tensorflow/regression_service.h"

namespace tensorflow {
namespace serving {

namespace {

int DeadlineToTimeoutMillis(const gpr_timespec deadline) {
  return gpr_time_to_millis(
      gpr_time_sub(gpr_convert_clock_type(deadline, GPR_CLOCK_MONOTONIC),
                   gpr_now(GPR_CLOCK_MONOTONIC)));
}

}  // namespace

SyntaxNetServiceImpl::SyntaxNetServiceImpl(ServerCore* core)
      : core_(core),
        parser_(new SyntaxNetParser()) {}

::grpc::Status SyntaxNetServiceImpl::Parse(::grpc::ServerContext* context,
                         const syntaxnet::SyntaxNetRequest* request,
                         syntaxnet::SyntaxNetResponse* response) {
  tensorflow::RunOptions run_options = tensorflow::RunOptions();
  run_options.set_timeout_in_ms(
      DeadlineToTimeoutMillis(context->raw_deadline()));

  const ::grpc::Status status =
      ToGRPCStatus(parser_->Parse(run_options, core_, *request, response));

  if (!status.ok()) {
    VLOG(1) << "Parse failed: " << status.error_message();
  }
  return status;
}

}  // namespace serving
}  // namespace tensorflow
