/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include "tensorflow_serving/servables/tensorflow/parser_impl.h"

#include "tensorflow/cc/saved_model/loader.h" // SavedModelBundle

namespace tensorflow {
namespace serving {

tensorflow::Status SyntaxNetParser::Parse(const tensorflow::RunOptions &run_options,
                         ServerCore *core,
                         const syntaxnet::SyntaxNetRequest &request,
                         syntaxnet::SyntaxNetResponse *response) {
  using namespace tensorflow;
  using namespace tensorflow::serving;
  if (!request.has_model_spec()) {
    return errors::InvalidArgument("Missing ModelSpec");
  }
  if (request.inputs().empty()) {
    return errors::InvalidArgument("expected at least one sentence");
  }
  // Validate signatures
  ServableHandle<SavedModelBundle> bundle;
  TF_RETURN_IF_ERROR(core->GetServableHandle(request.model_spec(), &bundle));
  const string signature_name = request.model_spec().signature_name();
  auto iter = bundle->meta_graph_def.signature_def().find(signature_name);
  if (iter == bundle->meta_graph_def.signature_def().end()) {
    return errors::FailedPrecondition(
              "Serving signature key not found.");
  }
  SignatureDef signature = iter->second;
  int sentences_count = request.inputs_size();
  Tensor inputs(tensorflow::DT_STRING, {sentences_count});
  for (int i = 0; i < sentences_count; i++) {
    if (request.inputs(i).token().empty()) {
      return errors::InvalidArgument("expected at least one token in a sentence");
    }
    inputs.vec<string>()(i) = request.inputs(i).SerializeAsString();
  }
  std::vector<Tensor> outputs;
  RunMetadata run_metadata;
  auto input_alias = "sentences";
  auto input_sig = signature.inputs().find(input_alias)->second;
  auto out_alias = "outputs";
  auto out_sig = signature.outputs().find(out_alias)->second;
  TF_RETURN_IF_ERROR(bundle->session->Run(
                     run_options,
                     {{input_sig.name(), inputs}},
                     {out_sig.name()},
                     {},
                     &outputs,
                     &run_metadata
  ));
//    LOG(INFO) << run_metadata.DebugString();
//    LOG(INFO) << outputs.size() << " output tensors available";
  auto sentences = outputs[0].vec<string>();
//    LOG(INFO) << "Sentences count: " << sentences.size();
  for (int i = 0; i < sentences.size(); ++i) {
    syntaxnet::Sentence *sentence = response->add_outputs();
    sentence->ParseFromString(sentences(i));
//      LOG(INFO) << "sentence " << i << ": " << sentence->DebugString();
  }
  return Status::OK();
}

}  // namespace serving
}  // namespace tensorflow