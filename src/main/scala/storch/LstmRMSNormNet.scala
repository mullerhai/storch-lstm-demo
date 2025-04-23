package storch

/*
 * Copyright 2022 storch.dev
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//> using scala "3.3"
//> using repository "sonatype:snapshots"
//> using repository "sonatype-s01:snapshots"
//> using lib "dev.storch::vision:0.0-2fff591-SNAPSHOT"
// replace with pytorch-platform-gpu if you have a CUDA capable GPU
//> using lib "org.bytedeco:pytorch-platform:2.1.2-1.5.10"
// enable for CUDA support
////> using lib "org.bytedeco:cuda-platform-redist:12.3-8.9-1.5.10"
// enable for native Apple Silicon support
// will not be needed with newer versions of pytorch-platform
////> using lib "org.bytedeco:pytorch:2.1.2-1.5.10,classifier=macosx-arm64"

//import example.FashionMNIST
import org.bytedeco.javacpp.{FloatPointer, PointerScope}
import org.bytedeco.pytorch.{OutputArchive, TensorExampleVectorIterator}
import torch.Device.{CPU, CUDA}
import torch.data.dataset.ChunkSharedBatchDataset
import torch.nn.functional as F
import torch.nn.modules.HasParams
import torch.optim.Adam
import torch.*
import torchvision.datasets.FashionMNIST

//import torchvision.datasets.FashionMNIST

import java.nio.file.Paths
//import scala.runtime.stdLibPatches.Predef.nn
import torch.internal.NativeConverters.{fromNative, toNative}

import scala.util.{Random, Using}

class LstmNet[D <: BFloat16 | Float32: Default](
    inputSize: Int = 28,
    hiddenSize: Int = 128,
    numLayers: Int = 2,
    numClasses: Int = 10
) extends HasParams[D] {

  val lstm = register(nn.LSTM(inputSize, hiddenSize, numLayers, batch_first = true))
  val fc = register(nn.Linear(hiddenSize, numClasses))
  val norm = register(nn.RMSNorm(Seq(28,hiddenSize)))

  def apply(i: Tensor[D]): Tensor[D] =
    val arr = Seq(numLayers, i.size.head, hiddenSize.toInt)
    val h0 = torch.zeros(size = arr, dtype = i.dtype)
    val c0 = torch.zeros(size = arr, dtype = i.dtype)
    val outTuple3 = lstm(i, Some(h0), Some(c0))

    var out: Tensor[D] = norm(outTuple3._1) // outTuple3._1
    out = out.index(torch.indexing.::, -1, ::)
    F.logSoftmax(fc(out), dim = 1)

}

class RnnNet[D <: BFloat16 | Float32: Default](
    inputSize: Int = 28,
    hiddenSize: Int = 128,
    numLayers: Int = 2,
    numClasses: Int = 10
) extends HasParams[D] {

  val rnn = register(nn.RNN(inputSize, hiddenSize, numLayers, batch_first = true))
  val fc = register(nn.Linear(hiddenSize, numClasses))
  val norm = register(nn.RMSNorm(Seq(28,hiddenSize)))
  def apply(i: Tensor[D]): Tensor[D] =
    val arr = Seq(numLayers, i.size.head, hiddenSize.toInt)
    val h0 = torch.zeros(size = arr, dtype = i.dtype)
    val c0 = torch.zeros(size = arr, dtype = i.dtype)
    val outTuple2 = rnn(i, Some(h0))
    var out: Tensor[D] = norm(outTuple2._1) //outTuple2._1
    out = out.index(torch.indexing.::, -1, ::)
    F.logSoftmax(fc(out), dim = 1)

}

class GruNet[D <: BFloat16 | Float32: Default](
    inputSize: Int = 28,
    hiddenSize: Int = 128,
    numLayers: Int = 2,
    numClasses: Int = 10
) extends HasParams[D] {

  val gru = register(nn.GRU(inputSize, hiddenSize, numLayers, batch_first = true))
  val fc = register(nn.Linear(hiddenSize, numClasses))
  val norm = register(nn.RMSNorm(Seq(28,hiddenSize)))
  def apply(i: Tensor[D]): Tensor[D] =
    val arr = Seq(numLayers, i.size.head, hiddenSize.toInt)
    val h0 = torch.zeros(size = arr, dtype = i.dtype)
    val c0 = torch.zeros(size = arr, dtype = i.dtype)
    val outTuple2 = gru(i, Some(h0))
    var out: Tensor[D] = norm(outTuple2._1) //outTuple2._1
    out = out.index(torch.indexing.::, -1, ::)
    F.logSoftmax(fc(out), dim = 1)

}

/** Shows how to train a simple LstmNet on the MNIST dataset */
object LstmRMSNormNet extends App {
  val device = if torch.cuda.isAvailable then CUDA else CPU
  println(s"Using device: $device")
//  val model = LstmNet().to(device)
//  val model = RnnNet().to(device)
  val model = LstmNet().to(device)
//  val model  = TransformerClassifier(embedding_dim = 128 , num_heads= 6, num_layers=6, hidden_dim = 30, num_classes=10, dropout_rate=0.1).to(device)
  // prepare data FashionMNIST
  //  val dataPath = Paths.get("data/mnist")
  //  val mnistTrain = MNIST(dataPath, train = true, download = true)
  //  val mnistEval = MNIST(dataPath, train = false)
  // "D:\\code\\data\\FashionMNIST"
//  val dataPath = Paths.get("data/FashionMNIST") //macos or linux
  val dataPath = Paths.get("D:\\data\\FashionMNIST") //windows
  val mnistTrain = FashionMNIST(dataPath, train = true, download = true)
  val mnistEval = FashionMNIST(dataPath, train = false)
  println(s"model ${model.modules.toSeq.mkString(" \n")}")
  println(s"model ${model.summarize}")
  val lossFn = torch.nn.loss.CrossEntropyLoss()
  // enable AMSGrad to avoid convergence issues
  val optimizer = Adam(model.parameters, lr = 1e-3, amsgrad = true)
  val optimizerCopy = Adam(model.parameters, lr = 1e-3, amsgrad = true)
  val evalFeatures = mnistEval.features.to(device)
  val evalTargets = mnistEval.targets.to(device)
  val r = Random(seed = 0)
  def dataLoader: Iterator[(Tensor[Float32], Tensor[Int64])] =
    r.shuffle(mnistTrain).grouped(8).map { batch =>
      val (features, targets) = batch.unzip
      (torch.stack(features).to(device), torch.stack(targets).to(device))
    }

  import org.bytedeco.pytorch.{ChunkDatasetOptions, Example, ExampleIterator, ExampleStack, ExampleVector, RandomSampler}
  import torch.data.DataLoaderOptions
  import torch.data.dataloader.ChunkRandomDataLoader
  import torch.data.datareader.ChunkDataReader
  import torch.data.dataset.{ChunkDataset, ChunkSharedBatchDataset}

  def exampleVectorToExample(exVec: ExampleVector): Example = {
    val example = new Example(exVec.get(0).data(), exVec.get(0).target())
    example
  }
//  val exampleTensorSeq = mnistTrain.map(x => new TensorExample(x._1.native))
//  val tensorExampleVector = new TensorExampleVector(exampleTensorSeq*)
//  val reader = new ChunkTensorDataReader()// new TensorExampleVectorReader()
//  reader(tensorExampleVector)

  val exampleSeq = mnistTrain.map(x => new Example(x._1.native, x._2.native))
//  val ex1 = new Example(mnistTrain.features.native ,mnistTrain.targets.native)
  val exampleVector = new ExampleVector(exampleSeq*)
  val reader = new ChunkDataReader()
//  {
//    override def read_chunk(chunk_index: Long) = exampleVector //  new ExampleVector(new Example(Tensor.create(10.0, 20.0, 50.0, 80.0, 100.0), Tensor.create(200.0)), new Example(Tensor.create(15.0, 30.0, 50.0, 80.0, 300.0), Tensor.create(400.0)), new Example(Tensor.create(20.0, 20.0, 50.0, 80.0, 100.0), Tensor.create(500.0)), new Example(Tensor.create(35.0, 30.0, 50.0, 80.0, 300.0), Tensor.create(600.0)), new Example(Tensor.create(40.0, 20.0, 50.0, 80.0, 100.0), Tensor.create(700.0)), new Example(Tensor.create(55.0, 30.0, 50.0, 80.0, 300.0), Tensor.create(800.0)), new Example(Tensor.create(60.0, 20.0, 50.0, 80.0, 100.0), Tensor.create(900.0)), new Example(Tensor.create(75.0, 30.0, 50.0, 80.0, 300.0), Tensor.create(300.0)))
//
//    override def chunk_count:Long = 1
//
//    override def reset(): Unit = {
//    }
//  }
  reader(exampleVector)
//
//  val ds = new JavaDataset() {
//     val exampleVector = new ExampleVector(exampleSeq.toArray:_*)
//    override def get(index: Long): Example = exampleVector.get(index)
//
//    override def size = new SizeTOptional(exampleVector.size)
//
//  }
//  val ds = new JD(reader)//.map(new ExampleStack())
//val ds = new JSD() {
//  val exampleVector = reader.exampleVec
//
//  override def get_batch(size: Long): ExampleVector = exampleVector
//
//  override def size = new SizeTOptional(exampleVector.size)
//}
//val ds = new TD() {
//  val tex = reader.tensorExampleVec //new TensorExampleVector(new TensorExample(Tensor.create(10.0, 20.0, 50.0, 80.0, 100.0)), new TensorExample(Tensor.create(15.0, 30.0, 50.0, 80.0, 300.0)), new TensorExample(Tensor.create(20.0, 20.0, 50.0, 80.0, 100.0)), new TensorExample(Tensor.create(35.0, 30.0, 50.0, 80.0, 300.0)), new TensorExample(Tensor.create(40.0, 20.0, 50.0, 80.0, 100.0)), new TensorExample(Tensor.create(55.0, 30.0, 50.0, 80.0, 300.0)), new TensorExample(Tensor.create(60.0, 20.0, 50.0, 80.0, 100.0)), new TensorExample(Tensor.create(75.0, 30.0, 50.0, 80.0, 300.0)))
//
//  override def get(index: Long): TensorExample = {
//    tex.get(index)
//    //                    return super.get(index);
//  }
//
//  override def get_batch(indices: SizeTArrayRef): TensorExampleVector = tex //.get_batch(indices) // ds.get_batch(indices) // exampleVector
//
//  override def size = new SizeTOptional(tex.size)
//}
  val batch_size = 32
  val prefetch_count = 1
//  val ds = new ChunkSharedBatchDataset(new ChunkDataset(reader, new RandomSampler(exampleSeq.size), new RandomSampler(exampleSeq.size), new ChunkDatasetOptions(prefetch_count, batch_size))).map(new ExampleStack)

  //  val ds  = new ChunkSharedTensorBatchDataset(new ChunkTensorDataset(reader,new RS(exampleTensorSeq.size),new ChunkDatasetOptions(prefetch_count, batch_size))).map(new TensorExampleStack)
  val ds = new ChunkSharedBatchDataset(
    new ChunkDataset(
      reader,
      new RandomSampler(exampleSeq.size),
      new RandomSampler(exampleSeq.size),
      new ChunkDatasetOptions(prefetch_count, batch_size)
    )
  ).map(new ExampleStack)

  //  val ds = new TensorDataset(reader)
//  val ds = new StreamDataset(reader)
  val opts = new DataLoaderOptions(32)
//  opts.workers.put(5)
  opts.batch_size.put(32)
//  opts.enforce_ordering.put(true)
//  opts.drop_last.put(false)
  val data_loader = new ChunkRandomDataLoader(ds, opts)
//  val data_loader = new ChunkRandomTensorDataLoader(ds, opts)
//  val data_loader = new JavaDistributedSequentialTensorDataLoader(ds, new DSS(ds.size.get), opts)
//  val data_loader = new JavaDistributedRandomTensorDataLoader(ds, new DRS(ds.size.get), opts)
//  val data_loader = new JavaSequentialTensorDataLoader(ds, new SS(ds.size.get), opts)
//  val data_loader = new JavaStreamDataLoader(ds, new STS(ds.size.get), opts)
//  val data_loader = new JavaStreamDataLoader(ds, new STS(ds.size.get), opts)
//  val data_loader = new JavaStreamDataLoader(ds, new StreamSampler(ds.size.get), opts)
//  val data_loader = new RandomDataLoader(ds, new RS(ds.size.get), opts)
//  val data_loader = new SequentialDataLoader(ds, new SS(ds.size.get), opts)
//  val data_loader = new DistributedSequentialDataLoader(ds, new DistributedSequentialSampler(ds.size.get), opts)
//  val data_loader = new DistributedRandomDataLoader(ds, new DistributedRandomSampler(ds.size.get), opts)
  //  val data_loader = new JavaRandomDataLoader(ds, new RandomSampler(ds.size.get), opts)
  println(s"ds.size.get {ds.size.get} data_loader option ${data_loader.options.batch_size()}")
  for (epoch <- 1 to 2) {
//    var it: ExampleVectorIterator = data_loader.begin
//    var it :TensorExampleVectorIterator = data_loader.begin
//    var it :TensorExampleIterator = data_loader.begin
    var it: ExampleIterator = data_loader.begin
    var batchIndex = 0
    println("coming in for loop")
    while (!it.equals(data_loader.end)) {
      Using.resource(new PointerScope()) { p =>
//        println(s"try to get loop data " )
        val batch = it.access
//        val es = new ExampleStack
//        val stack: Example = es.apply_batch(batch)
//        val tes = new TensorExampleStack
//        val stack: TensorExample = tes.apply_batch(batch)

//        println(s"get batch epoch ${epoch} batchIndex ${batchIndex} batch  ds.size.get {ds.size.get}   " )
        optimizer.zeroGrad()
        // println(s"epoch ${epoch} batchIndex ${batchIndex} batch ${batch.size} ds.size.get ${ds.size.get}   " )
        val trainDataTensor = fromNative(batch.data())
//        trainDataTensor.native.print()
        val prediction = model(fromNative(batch.data()).reshape(-1, 28, 28))
//        val loss = lossFn(prediction,fromNative(stack.target()))
        val loss = lossFn(prediction, fromNative(batch.target()))
        loss.backward()
        optimizer.step()
//        batch.size
//        prediction.native.print()
        //   println(s" epoch ${epoch} batchIndex ${batchIndex} ds.size.get ${ds.size.get}  prediction ${ prediction} + target + ${loss}")

//        System.out.println(s"${batch.get(0).data.createIndexer} + target + ${batch.get(0).target.createIndexer} +  first JavaDistributedSequentialTensorDataLoader ${batch.size} +  epoch: ${epoch}")
//        System.out.println(s"${batch.get(1).data.createIndexer}+  target  + ${batch.get(1).target.createIndexer} +  second JavaDistributedSequentialTensorDataLoader ${batch.size} +  epoch:   ${epoch}")
//        System.out.println(s"${batch.get(2).data.createIndexer}+ target  + ${batch.get(2).target.createIndexer} +  third JavaDistributedSequentialTensorDataLoader  ${batch.size} +  epoch:  + ${epoch}")
        it = it.increment
        batchIndex += 1
        if batchIndex % 200 == 0 then
          // run evaluation
          val predictions = model(evalFeatures.reshape(-1, 28, 28))
          val evalLoss = lossFn(predictions, evalTargets)
          val featuresData = new Array[Float](1000)
          val fp4 = new FloatPointer(predictions.native.data_ptr_float())
          fp4.get(featuresData)
          println(s"\n ffff size ${featuresData.size} shape ${evalFeatures.shape
              .mkString(", ")}a data ${featuresData.mkString(" ")}")
          println(s"predictions : ${predictions} \n")
          val accuracy =
            (predictions.argmax(dim = 1).eq(evalTargets).sum / mnistEval.length).item
          println(
            f"Epoch: $epoch | Batch: $batchIndex%4d | Training loss: ${loss.item}%.4f | Eval loss: ${evalLoss.item}%.4f | Eval accuracy: $accuracy%.4f"
          )
//        it = it.increment

      }
    }
    optimizerCopy.add_parameters(model.namedParameters()) //
    println(s"optimizerCopy ${optimizerCopy}")
    println(s"optimizer ${optimizer}")
    println(s"judge optimizer ${optimizer == optimizerCopy}")
    println(s"model parameters dict ${model.namedParameters()}")
  }

  // run training
//  for (epoch <- 1 to 50) do
//    for (batch <- dataLoader.zipWithIndex) do
//      // make sure we deallocate intermediate tensors in time shape [32,1,28,28]
//      Using.resource(new PointerScope()) { p =>
//        val ((feature, target), batchIndex) = batch
//        optimizer.zeroGrad()
//        val prediction = model(feature.reshape(-1,28,28))
//        val loss = lossFn(prediction, target)
//        loss.backward()
//        optimizer.step()
//        if batchIndex % 200 == 0 then
//          // run evaluation
//          val predictions= model(evalFeatures.reshape(-1,28,28))
//          val evalLoss = lossFn(predictions, evalTargets)
//          val featuresData  = new Array[Float](1000)
//          val fp4 = new FloatPointer(predictions.native.data_ptr_float())
//          fp4.get(featuresData)
//          println(s"\n ffff size ${featuresData.size} shape ${evalFeatures.shape.mkString(", ")}a data ${featuresData.mkString(" " )}")
//          println(s"predictions : ${predictions} \n")
//          val accuracy =
//            (predictions.argmax(dim = 1).eq(evalTargets).sum / mnistEval.length).item
//          println(
//            f"Epoch: $epoch | Batch: $batchIndex%4d | Training loss: ${loss.item}%.4f | Eval loss: ${evalLoss.item}%.4f | Eval accuracy: $accuracy%.4f"
//          )
//      }
    val archive = new OutputArchive
    model.save(archive)
    archive.save_to("net.pt")
}
