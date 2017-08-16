# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

from tensorflow_serving.apis import syntaxnet_service_pb2 as tensorflow__serving_dot_apis_dot_syntaxnet__service__pb2


class SyntaxNetServiceStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.Parse = channel.unary_unary(
        '/tensorflow.serving.SyntaxNetService/Parse',
        request_serializer=tensorflow__serving_dot_apis_dot_syntaxnet__service__pb2.SyntaxNetRequest.SerializeToString,
        response_deserializer=tensorflow__serving_dot_apis_dot_syntaxnet__service__pb2.SyntaxNetResponse.FromString,
        )


class SyntaxNetServiceServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def Parse(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_SyntaxNetServiceServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'Parse': grpc.unary_unary_rpc_method_handler(
          servicer.Parse,
          request_deserializer=tensorflow__serving_dot_apis_dot_syntaxnet__service__pb2.SyntaxNetRequest.FromString,
          response_serializer=tensorflow__serving_dot_apis_dot_syntaxnet__service__pb2.SyntaxNetResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'tensorflow.serving.SyntaxNetService', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
