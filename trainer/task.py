from trainer import image_classifier, augmentation_pipeline,GCSHelper
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def run(data_directory, output_directory, project_id,augment_flag,augment_samples,nr_epochs,drop_out,val_split,model,batch_size,check_overfit):
    image_classifier.check_input(project_id=project_id, data_dir=data_directory, output_dir=output_directory,
                                 validation_split=val_split, num_epochs=nr_epochs, dropout=drop_out,
                                 augmentation_samples=augment_samples)

    print('AUGMENTING IMAGES...')
    if augment_flag:
        augmentation_pipeline.augmentImages(project_id=project_id, data_dir=data_directory, sample_size=augment_samples,cloudML=True)
    print('AUGMENTING IMAGES DONE!')
    print('TRAINING MODEL...')
    image_classifier.retrain(project_id, data_directory, batch_size=batch_size, model=model, dropout=drop_out, num_epochs=nr_epochs,
                             validation_split=val_split, output_dir=output_directory, cloud_mode=True,check_overfit=check_overfit)




parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='directory of data')
parser.add_argument('--output_dir', type=str, help='directory of output model')
parser.add_argument('--project_id', type=str, default="trainer-cl", help='Google cloud projectID')
parser.add_argument('--aug_flag', type=str2bool, default=False, help='True if augmentation is done on images')
parser.add_argument('--aug_samples', type=int, default=1, help='extra augmentation samples that are added per category')
parser.add_argument('--nr_epochs', type=int, default=1, help='extra augmentation samples that are added per category')
parser.add_argument('--drop_out', type=float, default=0.1, help='Amount of droppout to prevent overfitting')
parser.add_argument('--val_split', type=float, default=0.1, help='Percentage of data used for validation')
parser.add_argument('--model', type=str, default="MobileNet", help='Used model architecture')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size used for model training')
parser.add_argument('--check_overfit', type=str2bool, default=True, help='Add early stopping check')




args = parser.parse_args()

try:
    run(args.data_dir,args.output_dir,args.project_id,args.aug_flag,args.aug_samples,args.nr_epochs,args.drop_out,args.val_split,args.model,args.batch_size,args.check_overfit)
    GCSHelper.uploadClosingStatusFilesToGCS(args.project_id,[],'done.txt',args.output_dir)
except Exception as e:
    GCSHelper.uploadClosingStatusFilesToGCS(args.project_id,[str(e)], 'wrong.txt', args.output_dir)




